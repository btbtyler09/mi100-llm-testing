"""Keep multi-request GLM prefills within the bounded indexer workspace."""
from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch

from vllm.v1.attention.backends.mla import indexer


@pytest.mark.parametrize('concurrency', [1, 2, 4])
def test_glm_workspace_stays_bounded(monkeypatch, concurrency):
    monkeypatch.setattr(indexer.current_platform, 'is_rocm', lambda: True)
    rocm = ModuleType('vllm.platforms.rocm')
    rocm.on_gfx908 = lambda: True
    monkeypatch.setitem(sys.modules, 'vllm.platforms.rocm', rocm)
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=4096,
            hf_config=SimpleNamespace(model_type='glm5_next')),
        scheduler_config=SimpleNamespace(max_num_seqs=concurrency),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1,
                                        prefill_context_parallel_size=1))
    assert indexer.get_max_prefill_buffer_size(config) == 4096


@pytest.mark.parametrize('compression', [1, 4])
def test_four_full_contexts_are_chunked_without_overflow(compression):
    workspace = 4096
    lengths = torch.tensor([4096 // compression] * 4)
    queries = torch.tensor([64, 128, 256, 512])
    chunks = indexer.DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks(
        lengths, queries, workspace, max_logits_bytes=2**30)
    covered = []
    for reqs, query in chunks:
        assert int(lengths[reqs].sum()) <= workspace
        assert query.start == 0
        assert query.stop == int(queries[reqs].sum())
        covered.extend(range(reqs.start, reqs.stop))
    assert covered == [0, 1, 2, 3]
