# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# AITER formulation copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Conservative GLM mHC kernels for gfx908.

Keep FP32 projection/norm arithmetic and explicit BF16 rounding boundaries;
opt-in small-token paths fuse projection and pre/RMSNorm. The Sinkhorn order follows mhc_pre_torch
and AITER's asymmetric exp-domain formulation (MIT, AMD, 2024-2026):
https://github.com/ROCm/aiter/blob/bcb56d9c4d04bb4957a051911da69162cb21c3af/aiter/ops/triton/_triton_kernels/fusions/mhc.py
This implementation uses the local vLLM parameter/layout contract.
"""
import os
import torch

from vllm.triton_utils import tl, triton

PROJECT_DECODE = os.environ.get("GLM53_GFX908_MHC_PROJECT", "0") == "1"
FUSE_PRE_NORM = os.environ.get("GLM53_GFX908_MHC_FUSE_NORM", "0") == "1"


@triton.jit
def _project_kernel(Residual, Fn, Mixes, H: tl.constexpr, RMS_EPS: tl.constexpr):
    token, row = tl.program_id(0), tl.program_id(1)
    k = tl.arange(0, 4 * H)
    x = tl.load(Residual + token * 4 * H + k).to(tl.float32)
    weight = tl.load(Fn + row * 4 * H + k)
    projection = tl.sum(x * weight, 0)
    inverse_rms = tl.rsqrt(tl.sum(x * x, 0) / (4 * H) + RMS_EPS)
    tl.store(Mixes + token * 24 + row, projection * inverse_rms)


@triton.jit
def _apply_pre_norm_kernel(Residual, Pre, Weight, Output,
                          H: tl.constexpr, EPS: tl.constexpr,
                          ROUND_BEFORE_WEIGHT: tl.constexpr):
    token = tl.program_id(0)
    h, streams = tl.arange(0, H), tl.arange(0, 4)
    residual = tl.load(Residual + token * 4 * H + streams[:, None] * H + h[None, :]).to(tl.float32)
    weights = tl.load(Pre + token * 4 + streams)
    # This explicit round trip preserves the reference BF16 boundary.
    x = tl.sum(weights[:, None] * residual, 0).to(tl.bfloat16).to(tl.float32)
    inverse_rms = tl.rsqrt(tl.sum(x * x, 0) / H + EPS)
    norm_weight = tl.load(Weight + h).to(tl.float32)
    normalized = x * inverse_rms
    if ROUND_BEFORE_WEIGHT:
        normalized = normalized.to(tl.bfloat16).to(tl.float32)
    tl.store(Output + token * H + h, normalized * norm_weight)


@triton.jit
def _mix_kernel(Mixes, Scale, Base, Pre, Post, Comb,
                PRE_EPS: tl.constexpr, SK_EPS: tl.constexpr,
                POST_MULT: tl.constexpr, ITERS: tl.constexpr):
    token = tl.program_id(0)
    i = tl.arange(0, 4)
    pre_logits = tl.load(Mixes + token * 24 + i) * tl.load(Scale) + tl.load(Base + i)
    pre = tl.sigmoid(pre_logits) + PRE_EPS
    post_logits = (tl.load(Mixes + token * 24 + 4 + i) * tl.load(Scale + 1)
                   + tl.load(Base + 4 + i))
    tl.store(Pre + token * 4 + i, pre)
    tl.store(Post + token * 4 + i, tl.sigmoid(post_logits) * POST_MULT)
    j = i[:, None] * 4 + i[None, :]
    logits = (tl.load(Mixes + token * 24 + 8 + j) * tl.load(Scale + 2)
              + tl.load(Base + 8 + j))
    values = tl.exp(logits - tl.max(logits, 1)[:, None])
    values = values / tl.sum(values, 1)[:, None] + SK_EPS
    values = values / (tl.sum(values, 0)[None, :] + SK_EPS)
    for _ in range(ITERS - 1):
        values = values / (tl.sum(values, 1)[:, None] + SK_EPS)
        values = values / (tl.sum(values, 0)[None, :] + SK_EPS)
    tl.store(Comb + token * 16 + j, values)


@triton.jit
def _apply_pre_kernel(Residual, Pre, Output, H: tl.constexpr, BLOCK: tl.constexpr):
    token, block = tl.program_id(0), tl.program_id(1)
    h = block * BLOCK + tl.arange(0, BLOCK)
    streams = tl.arange(0, 4)
    residual = tl.load(Residual + token * 4 * H + streams[:, None] * H + h[None, :],
                       h[None, :] < H, other=0).to(tl.float32)
    weights = tl.load(Pre + token * 4 + streams)
    layer_input = tl.sum(weights[:, None] * residual, 0)
    tl.store(Output + token * H + h, layer_input, h < H)


@triton.jit
def _post_kernel(X, Residual, Post, Comb, Output, H: tl.constexpr, BLOCK: tl.constexpr):
    token, dest, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    h = block * BLOCK + tl.arange(0, BLOCK)
    src = tl.arange(0, 4)
    values = tl.load(Residual + token * 4 * H + src[:, None] * H + h[None, :],
                     h[None, :] < H, other=0).to(tl.float32)
    weights = tl.load(Comb + token * 16 + src * 4 + dest)
    mixed = tl.sum(values * weights[:, None], 0)
    x = tl.load(X + token * H + h, h < H, other=0).to(tl.float32)
    post = tl.load(Post + token * 4 + dest)
    tl.store(Output + token * 4 * H + dest * H + h, mixed + post * x, h < H)


def supported(residual, fn=None):
    return (residual.dtype == torch.bfloat16 and residual.ndim >= 3
            and residual.shape[-2] == 4
            and (fn is None or fn.dtype == torch.float32))


def pre(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
        hc_post_mult_value, sinkhorn_repeat, norm_weight=None, norm_eps=0.0):
    outer = residual.shape[:-2]
    hidden = residual.shape[-1]
    residual = residual.reshape(-1, 4, hidden).contiguous()
    tokens = residual.shape[0]
    if PROJECT_DECODE and hidden == 4096 and 0 < tokens <= 4 and fn.is_contiguous():
        mixes = torch.empty((tokens, 24), device=residual.device, dtype=torch.float32)
        _project_kernel[(tokens, 24)](residual, fn, mixes, hidden, rms_eps,
            num_warps=4, enable_fp_fusion=False)
    else:
        # Retain the accepted projection for prefill and unsupported shapes.
        x = residual.view(tokens, 4 * hidden).float()
        mixes = torch.matmul(x, fn.t())
        sqrsum = x.square().sum(dim=-1, keepdim=True)
        mixes = mixes * torch.rsqrt(sqrsum / (4 * hidden) + rms_eps)
    pre_mix = torch.empty((tokens, 4), device=residual.device, dtype=torch.float32)
    post_mix = torch.empty_like(pre_mix)
    comb_mix = torch.empty((tokens, 4, 4), device=residual.device, dtype=torch.float32)
    output = torch.empty((tokens, hidden), device=residual.device, dtype=residual.dtype)
    _mix_kernel[(tokens,)](mixes, hc_scale, hc_base, pre_mix, post_mix, comb_mix,
        hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
        num_warps=1, enable_fp_fusion=False)
    norm_provider = None
    if (FUSE_PRE_NORM and norm_weight is not None and hidden == 4096
            and 0 < tokens <= 4 and norm_weight.dtype == torch.bfloat16):
        from vllm import ir
        norm_provider = ir.ops.rms_norm.dispatch(output, norm_weight, norm_eps).provider
    if norm_provider in ("native", "vllm_c"):
        # Native IR rounds normalized x to weight.dtype before multiplying;
        # vllm_c multiplies in FP32 and rounds only the final result.
        _apply_pre_norm_kernel[(tokens,)](residual, pre_mix, norm_weight, output,
            hidden, norm_eps, norm_provider == "native",
            num_warps=4, enable_fp_fusion=False)
    else:
        _apply_pre_kernel[(tokens, triton.cdiv(hidden, 256))](
            residual, pre_mix, output, hidden, 256, num_warps=4, enable_fp_fusion=False)
        # Retain the BF16 stream-reduction rounding before the accepted RMSNorm.
        if norm_weight is not None:
            from vllm import ir
            output = ir.ops.rms_norm(output, norm_weight, norm_eps)
    return (post_mix.view(*outer, 4, 1), comb_mix.view(*outer, 4, 4),
            output.view(*outer, hidden))


def post(x, residual, post_mix, comb_mix):
    outer = residual.shape[:-2]
    hidden = residual.shape[-1]
    residual = residual.reshape(-1, 4, hidden).contiguous()
    x = x.reshape(-1, hidden).contiguous()
    output = torch.empty_like(residual)
    _post_kernel[(residual.shape[0], 4, triton.cdiv(hidden, 256))](
        x, residual, post_mix.contiguous(), comb_mix.contiguous(), output,
        hidden, 256, num_warps=4, enable_fp_fusion=False)
    return output.view(*outer, 4, hidden)


def fused_post_pre(x, residual, post_mix, comb_mix, fn, hc_scale, hc_base,
                   rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
                   sinkhorn_repeat, norm_weight=None, norm_eps=0.0):
    residual_cur = post(x, residual, post_mix, comb_mix)
    return (residual_cur, *pre(residual_cur, fn, hc_scale, hc_base, rms_eps,
        hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
        norm_weight, norm_eps))
