"""Allocation/release and real buffer zeroing for pooled hybrid caches."""
from types import SimpleNamespace
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import KpoolTailManager
from vllm.v1.kv_cache_interface import KpoolTailSpec, MLAAttentionSpec
from vllm.v1.worker.utils import AttentionGroup, KVBlockZeroer


def test_tail_allocation_free_and_cancel_reuse():
    pool = BlockPool(8, enable_caching=False, hash_block_size=4)
    spec = KpoolTailSpec(block_size=4, num_kv_heads=2, head_size=128,
                         head_size_v=0, dtype=torch.bfloat16, sliding_window=4)
    manager = KpoolTailManager(kv_cache_spec=spec, block_pool=pool,
        enable_caching=False, kv_cache_group_id=0, scheduler_block_size=640)
    for cycle in range(20):
        a, b = f'a{cycle}', f'b{cycle}'
        first = manager.allocate_new_blocks(a, 3, 3)
        second = manager.allocate_new_blocks(b, 5, 5)
        assert len(first) == len(second) == 1
        assert first[0].block_id != second[0].block_id
        for n in (4, 5, 63, 64, 65, 639, 640, 641, 8192):
            assert manager.allocate_new_blocks(a, n, n) == []
        manager.free(a)  # cancellation follows the same release path
        assert first[0].ref_cnt == 0 and second[0].ref_cnt == 1
        manager.free(b)
        assert pool.get_num_free_blocks() == 7  # block zero is the null block


def test_zeroer_uses_pooled_view_geometry_with_shared_allocation():
    blocks, page = 3, 128 * 512 * 2
    raw = torch.full((blocks, page), 17, dtype=torch.uint8, device='cuda')
    main = raw.view(torch.bfloat16).view(blocks * 2, 64, 512)
    pooled = torch.as_strided(raw, (blocks, 32, 132), (page, 132, 1))
    specs = [
        MLAAttentionSpec(block_size=128, num_kv_heads=1, head_size=512,
                         dtype=torch.bfloat16),
        MLAAttentionSpec(block_size=128, num_kv_heads=1, head_size=132,
                         dtype=torch.uint8, storage_block_size=32),
    ]
    groups = [AttentionGroup(None, [name], spec, i)
              for i, (name, spec) in enumerate(zip(('main', 'pooled'), specs))]
    zeroer = KVBlockZeroer(torch.device('cuda'), groups, [64, 64],
        {'main': SimpleNamespace(kv_cache=main), 'pooled': SimpleNamespace(kv_cache=pooled)},
        num_blocks=blocks)
    zeroer.zero_block_ids([[1], [1]])
    assert torch.all(raw[1] == 0)
    assert torch.all(raw[0] == 17) and torch.all(raw[2] == 17)


def test_zeroer_does_not_clear_another_groups_live_pages():
    blocks = 4
    caches = [torch.full((blocks, 64, 512), 7, device='cuda', dtype=torch.bfloat16),
              torch.full((blocks, 32, 132), 19, device='cuda', dtype=torch.uint8)]
    specs = [MLAAttentionSpec(block_size=64, num_kv_heads=1, head_size=512,
                              dtype=torch.bfloat16),
             MLAAttentionSpec(block_size=128, num_kv_heads=1, head_size=132,
                              dtype=torch.uint8, storage_block_size=128)]
    groups = [AttentionGroup(None, [str(i)], spec, i) for i, spec in enumerate(specs)]
    zeroer = KVBlockZeroer(torch.device('cuda'), groups, [64, 64],
        {str(i): SimpleNamespace(kv_cache=cache) for i, cache in enumerate(caches)},
        num_blocks=blocks)
    zeroer.zero_block_ids([[1], []])
    assert torch.all(caches[0][1] == 0)
    assert torch.all(caches[1] == 19)
    zeroer.zero_block_ids([[], [2]])
    assert torch.all(caches[1][2] == 0)
    assert torch.all(caches[0][2] == 7)
