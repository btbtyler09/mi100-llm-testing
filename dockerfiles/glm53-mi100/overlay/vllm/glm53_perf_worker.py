# SPDX-License-Identifier: Apache-2.0
"""Explicit worker extension for the loopback-only eager performance canary.

Not configured in production. Changes are made only between completed requests.
"""
import json
import os
from contextlib import contextmanager

import torch


@contextmanager
def _moe_config(config):
    import vllm.model_executor.layers.fused_moe as moe
    previous = moe._config
    try:
        moe._config = config
        yield
    finally:
        moe._config = previous


class GLM53PerfWorker:
    def glm53_set_variant(self, settings: str):
        data = json.loads(settings)
        assert not set(data) - {"mhc", "validate_logits", "threads", "moe_config",
                                "mhc_project", "mhc_norm", "profile_tables", "router",
                                "clear_profiler"}
        if not self.model_config.enforce_eager:
            # Logits remain outside capture unless explicitly requested by a
            # separate base-image feature. CPU pool sizes affect eager host work,
            # not the arithmetic or addresses already recorded in a GPU graph.
            # Captured model/kernel settings remain immutable.
            assert not set(data) - {"validate_logits", "profile_tables", "clear_profiler", "threads"}, "Do not mutate captured graph settings"
            assert os.environ.get("VLLM_GFX908_LOGITS_IN_GRAPH", "0") != "1"
        from vllm.model_executor.layers import mhc
        if "mhc" in data:
            assert isinstance(data["mhc"], bool)
            mhc.USE_GFX908_MHC = data["mhc"]
        if "validate_logits" in data:
            assert isinstance(data["validate_logits"], bool)
            os.environ["GLM53_VALIDATE_LOGITS"] = "1" if data["validate_logits"] else "0"
        if "threads" in data:
            assert data["threads"] in (1, 2, 4)
            torch.set_num_threads(data["threads"])
        from vllm.model_executor.kernels.mhc import gfx908
        for key, attribute in (("mhc_project", "PROJECT_DECODE"), ("mhc_norm", "FUSE_PRE_NORM")):
            if key in data:
                assert isinstance(data[key], bool)
                setattr(gfx908, attribute, data[key])
        if "profile_tables" in data:
            assert isinstance(data["profile_tables"], bool)
            self.profiler_config.torch_profiler_dump_cuda_time_total = data["profile_tables"]
        if data.get("clear_profiler"):
            assert self.profiler is None or (not self.profiler._active and not self.profiler._running)
            self.profiler = None
            import gc
            gc.collect()
        from vllm.models.glm5next.nvidia import gfx908_router
        if "router" in data:
            assert isinstance(data["router"], bool)
            gfx908_router.USE_ROUTER = data["router"]
        if "moe_config" in data:
            import vllm.model_executor.layers.fused_moe as moe
            moe._config = data["moe_config"] or None
        return {"rank": self.rank, "mhc": mhc.USE_GFX908_MHC,
            "mhc_project": gfx908.PROJECT_DECODE, "mhc_norm": gfx908.FUSE_PRE_NORM,
            "router": gfx908_router.USE_ROUTER,
            "validate_logits": os.environ.get("GLM53_VALIDATE_LOGITS"),
            "torch_threads": torch.get_num_threads(),
            "allocated_gib": torch.cuda.memory_allocated() / 2**30}

    @torch.inference_mode()
    def glm53_moe_probe(self, settings: str):
        """Time a resident actual-weight layer, using all ranks and its TP reduce."""
        assert self.model_config.enforce_eager
        data = json.loads(settings)
        from vllm.forward_context import set_forward_context
        model = self.model_runner.get_model()
        layer = model.language_model.model.layers[3].mlp
        count = int(data.get("tokens", 1))
        assert count in (1, 2, 4, 128, 512)
        torch.manual_seed(101)
        x = torch.randn(count, 4096, device=self.device, dtype=torch.bfloat16) * 0.1
        with set_forward_context(None, self.vllm_config, num_tokens=count):
            with _moe_config(None):
                reference = layer(x).clone()
            with _moe_config(data.get("config")):
                for _ in range(3):
                    result = layer(x)
                torch.cuda.synchronize()
                # Cache-flushed weight microbenchmarks are handled separately;
                # this probe screens launch/resource choices with resident weights.
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(20):
                    result = layer(x)
                end.record()
                end.synchronize()
        difference = (result.float() - reference.float())
        return {"rank": self.rank, "tokens": count, "config": data.get("config"),
            "ms": start.elapsed_time(end) / 20,
            "relative_l2": (difference.norm() / reference.float().norm()).item(),
            "finite": bool(torch.isfinite(result).all())}
