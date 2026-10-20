# SPDX-License-Identifier: Apache-2.0
"""MI100 indexer helpers for the unshuffled e4m3fn pooled cache."""
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _gather_cache(cache, values, scales, table, cumulative, token_to_seq,
                  CACHE_STRIDE: tl.constexpr, TABLE_STRIDE: tl.constexpr,
                  BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr,
                  VALUE_STRIDE: tl.constexpr, SCALE_STRIDE: tl.constexpr):
    row = tl.program_id(0)
    seq = tl.load(token_to_seq + row)
    start = tl.load(cumulative + seq)
    pos = row - start
    block = tl.load(table + seq * TABLE_STRIDE + pos // BLOCK_SIZE)
    slot = pos % BLOCK_SIZE
    col = tl.arange(0, triton.next_power_of_2(HEAD_DIM))
    v = tl.load(cache + block * CACHE_STRIDE + slot * HEAD_DIM + col,
                col < HEAD_DIM, 0)
    tl.store(values + row * VALUE_STRIDE + col, v, col < HEAD_DIM)
    byte = tl.arange(0, 4)
    s = tl.load(cache + block * CACHE_STRIDE + BLOCK_SIZE * HEAD_DIM + slot * 4 + byte)
    tl.store(scales + row * SCALE_STRIDE + byte, s)


def gather_indexer_cache(cache, values, scales, table, cumulative, token_to_seq):
    """Gather quantized bytes without ROCm's newer-GPU preshuffle layout."""
    assert cache.dtype == torch.uint8
    assert values.dtype == torch.float8_e4m3fn
    assert values.is_contiguous() and scales.is_contiguous()
    _gather_cache[(values.shape[0],)](
        cache, values.view(torch.uint8), scales.view(torch.uint8),
        table, cumulative, token_to_seq,
        cache.stride(0), table.stride(0), cache.shape[1], values.shape[1],
        values.stride(0), scales.stride(0) * scales.element_size(),
    )
