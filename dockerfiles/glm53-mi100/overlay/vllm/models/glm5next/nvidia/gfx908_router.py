# SPDX-License-Identifier: Apache-2.0
"""GLM-shaped BF16 router GEMV with FP32 output on MI100."""
import os
import torch
from vllm.model_executor.layers.fused_moe import GateLinear
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

USE_ROUTER = os.environ.get('GLM53_GFX908_ROUTER', '0') == '1'


@triton.jit
def _router_kernel(X, Weight, Output, XS0: tl.constexpr, XS1: tl.constexpr):
    token, expert = tl.program_id(0), tl.program_id(1)
    k = tl.arange(0, 4096)
    x = tl.load(X + token * XS0 + k * XS1).to(tl.float32)
    weight = tl.load(Weight + expert * 4096 + k).to(tl.float32)
    tl.store(Output + token * 288 + expert, tl.sum(x * weight, 0))


def router_gemv(x, weight):
    output = torch.empty((x.shape[0], 288), dtype=torch.float32, device=x.device)
    _router_kernel[(x.shape[0], 288)](x, weight, output, *x.stride(),
        num_warps=4, enable_fp_fusion=False)
    return output


class Glm53GateLinear(GateLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._glm53_gfx908 = False
        if current_platform.is_rocm():
            from vllm.platforms.rocm import on_gfx908
            self._glm53_gfx908 = on_gfx908()

    def forward(self, x):
        if (USE_ROUTER and self._glm53_gfx908 and self.bias is None
                and self.out_dtype == torch.float32
                and x.ndim == 2 and 0 < x.shape[0] <= 4
                and x.dtype == self.weight.dtype == torch.bfloat16
                and self.weight.shape == (288, 4096) and self.weight.is_contiguous()):
            return router_gemv(x, self.weight), None
        return super().forward(x)
