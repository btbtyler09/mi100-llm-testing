# SPDX-License-Identifier: Apache-2.0
"""Selected-pair gfx908 INT4 GEMV for the GLM TP8 M=1/2 expert shapes.

BF16 dequantized operands, FP32 partial sums, and BF16 GEMM boundaries match
Triton WNA16. Callers keep the existing activation and expert-sum providers.
The original packed weights are read directly; there is no duplicate layout.
"""
import torch
from vllm import _custom_ops as ops
from vllm.triton_utils import triton, tl

@triton.jit
def _gemv(A,W,S,Ids,Routing,P,N:tl.constexpr,K:tl.constexpr,SPLITS:tl.constexpr,
          SHARED_A:tl.constexpr,MUL_WEIGHT:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    row=tl.program_id(0);nb=tl.program_id(1);part=tl.program_id(2)
    ns=nb*BN+tl.arange(0,BN)
    ks=part*BK+tl.arange(0,BK)
    expert=tl.load(Ids+row)
    ar=row//8 if SHARED_A else row
    a=tl.load(A+ar*K+ks,mask=ks<K,other=0).to(tl.float32)
    packed=tl.load(W+expert*N*(K//2)+ns[:,None]*(K//2)+ks[None,:]//2,
                   mask=(ns[:,None]<N)&(ks[None,:]<K),other=0)
    scale=tl.load(S+expert*N*(K//128)+ns[:,None]*(K//128)+ks[None,:]//128,
                  mask=(ns[:,None]<N)&(ks[None,:]<K),other=0).to(tl.float32)
    q=((packed.to(tl.int32)>>((ks[None,:]%2)*4))&15)-8
    w=(q.to(tl.float32)*scale).to(tl.bfloat16).to(tl.float32)
    val=tl.sum(w*a[None,:],axis=1)
    if MUL_WEIGHT:
        val=val*tl.load(Routing+row)
    tl.store(P+(row*N+ns)*SPLITS+part,val,mask=ns<N)


@triton.jit
def _finish_projection(P,Out,COUNT:tl.constexpr,SPLITS:tl.constexpr,B:tl.constexpr):
    rows=tl.program_id(0)*B+tl.arange(0,B)
    ss=tl.arange(0,SPLITS)
    v=tl.load(P+rows[:,None]*SPLITS+ss[None,:],mask=rows[:,None]<COUNT,other=0.)
    tl.store(Out+rows,tl.sum(v,axis=1),mask=rows<COUNT)


def direct_w4(output, hidden, w1, w2, s1, s2, weights, ids,
              workspace13, workspace2, activate):
    m=hidden.shape[0];rows=m*8
    # The existing workspaces have M*8*4096 BF16 elements. Up-projection
    # partials consume exactly that many bytes (512 outputs * 4 FP32 splits).
    # Complete each producer before reusing its scratch for the next stage.
    partial=workspace2.view(torch.float32).flatten()[:rows*512*4].view(rows,512,4)
    projected=workspace13.flatten()[:rows*512].view(rows,512)
    h=torch.empty((rows,256),device=hidden.device,dtype=torch.bfloat16)
    down=workspace2.flatten()[:rows*4096].view(m,8,4096)
    _gemv[(rows,64,4)](hidden,w1,s1,ids,weights,partial,512,4096,4,True,False,8,1024,num_warps=4)
    _finish_projection[(triton.cdiv(rows*512,128),)](partial,projected,rows*512,4,128,num_warps=4)
    activate(h,projected)
    _gemv[(rows,512,1)](h,w2,s2,ids,weights,down,4096,256,1,False,True,8,256,num_warps=4)
    ops.moe_sum(down,output)
