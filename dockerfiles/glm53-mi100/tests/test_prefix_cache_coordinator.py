"""Exercise the actual coordinator/manager boundary and finite cache lifecycle."""
from types import SimpleNamespace

import torch

from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec, KpoolTailSpec, KVCacheConfig, KVCacheGroupSpec,
)


def make_coordinator():
    full = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=8,
                             dtype=torch.bfloat16)
    tail = KpoolTailSpec(block_size=4, num_kv_heads=2, head_size=8,
                        head_size_v=0, dtype=torch.bfloat16, sliding_window=4)
    config = KVCacheConfig(num_blocks=9, kv_cache_tensors=[], kv_cache_groups=[
        KVCacheGroupSpec(['attention'], full), KVCacheGroupSpec(['tail'], tail)])
    return HybridKVCacheCoordinator(config, max_model_len=64,
        max_in_flight_tokens=16, use_eagle=False, enable_caching=True,
        enable_kv_cache_events=False, dcp_world_size=1, pcp_world_size=1,
        scheduler_block_size=4, hash_block_size=4)


def test_cache_lookup_release_eviction_and_reset():
    coord = make_coordinator()
    pool = coord.block_pool
    for cycle in range(12):
        hashes = [(cycle * 10 + n).to_bytes(32, 'big') for n in (1, 2, 3)]
        request = SimpleNamespace(request_id=str(cycle), num_prompt_tokens=12,
                                  shared_prefix_boundary=0, block_hashes=hashes)
        coord.allocate_new_blocks(request.request_id, 12, 12)
        coord.cache_blocks(request, 12)
        assert all(m.cache_hit_alignment_tokens == 4 for m in coord.single_type_managers)
        coord.free(request.request_id)
        assert pool.get_num_free_blocks() == 8
        _, hit_tokens, _ = coord.find_longest_cache_hit(hashes, 11)
        assert hit_tokens == 8
    assert pool.reset_prefix_cache()
    _, hit_tokens, _ = coord.find_longest_cache_hit(hashes, 11)
    assert hit_tokens == 0


def test_replay_boundaries_include_resend_and_extension():
    coord = make_coordinator()
    request = SimpleNamespace(num_prompt_tokens=12)
    assert coord.get_replay_boundaries(request) == (11,)
    coord.eagle_group_ids = {0}
    assert coord.get_replay_boundaries(request) == (4, 8)


def test_hybrid_prefill_stops_at_reusable_mamba_boundaries():
    from vllm.v1.core.sched.scheduler import Scheduler

    scheduler = SimpleNamespace(cache_config=SimpleNamespace(block_size=4),
        block_size=640, hash_block_size=640, use_eagle=False,
        mamba_has_prefill_checkpoint_blocks=False, mamba_partial_cache_hit=False,
        max_num_scheduled_tokens=2048,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0))
    request = SimpleNamespace(num_tokens=4096, num_prompt_tokens=4096,
                              num_computed_tokens=0, shared_prefix_boundary=0)
    ends = []
    while request.num_computed_tokens < request.num_prompt_tokens:
        n = Scheduler._mamba_block_aligned_split(scheduler, request,
            min(2048, request.num_prompt_tokens - request.num_computed_tokens))
        assert n > 0
        request.num_computed_tokens += n
        ends.append(request.num_computed_tokens)
    # The old code ended at [2048, 4096], never materializing the reusable
    # state at 3840. Its only final state lay past the last reusable token.
    assert ends == [1920, 3840, 4096]
