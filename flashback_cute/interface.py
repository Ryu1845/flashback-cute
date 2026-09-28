"""Flashback (softmax-attention double backward) on top of FlashAttention-4, SM90.

Forward and first-order backward are stock FA4 kernels. The double backward (grads of
(dq, dk, dv) w.r.t. (q, k, v, dout)) runs two CuTe-DSL kernels derived from FA4's SM90
forward / backward:

  K1 FlashbackJvpFwdSm90  (Q-block outer): dot_dout = (P*Sdot) V + P gv - z1*O,  z1 = rowsum(P*Sdot),
                          Ddot = rowsum(dO*dot_dout)
  K2 FlashbackHvpBwdSm90  (KV-block outer): dot_q, dot_k, dot_v

with Sdot = c (Q gk^T + gq K^T), Pdot = P (Sdot - z1), D = rowsum(dO*O),
dS = P (dO V^T - D), dSdot = Pdot (dO V^T - D) + P (dO gv^T - Ddot),
dot_v = Pdot^T dO, dot_k = c (dSdot^T Q + dS^T gq), dot_q = c (dSdot K + dS gk).
"""

from typing import Optional

import torch

import cutlass.cute as cute
from flash_attn.cute.interface import (
    _flash_attn_fwd,
    _flash_attn_bwd,
    _bwd_preprocess,
    _bwd_postprocess_convert,
    torch2cute_dtype_map,
)
from flash_attn.cute.cute_dsl_utils import to_cute_tensor

from flashback_cute.jvp_fwd_sm90 import FlashbackJvpFwdSm90
from flashback_cute.hvp_bwd_sm90 import FlashbackHvpBwdSm90


K1_TILE_M = 128
K1_TILE_N = 64
K1_NUM_STAGES = 2
# K2 tile_m is shared with K1 (tile_m_bwd), preprocess and postprocess: it is the row padding
# granularity of the fp32 stats / dot_q accumulator buffers.
K2_TILE_M = 64
# Per head_dim K2 (FlashbackHvpBwdSm90) MMA configs. Register budget (240/thread) is what
# decides: 4 S-like fp32 accumulators + dot_k/dot_v accumulators + dot_q accumulator.
K2_CONFIGS = {
    # 64x128 tile, S-like GEMMs transposed so Pdot^T / dS^T / dSdot^T feed dot_v / dot_k
    # from registers (FA4's hdim-128 layout); 8 of 10 GEMMs run at full WGMMA width.
    64: dict(tile_n=128, Q_stage=2, PdS_stage=2, SdP_swapAB=True, AtomLayoutNdKV=2),
    # 64x128 would need 288 accumulator regs/thread (dot_k + dot_v alone take 128), so the
    # tile stays 64x64. Two PdS stages drop the cross-WG barrier before the Pdot store
    # (0.7-2% faster non-causal, 3-5% causal); their 24 KB of smem is paid for by a single dO
    # stage (the pair lands just under the 227 KB limit). Measured slower on this config: also
    # dropping the barrier before the dS store (g6 then waits for the post-store barrier;
    # ~2% slower causal), Q / dO as register A operands of the S-like GEMMs (1-7%), and
    # red.global dot_q accumulation instead of smem + TMA reduce (7-9%).
    128: dict(tile_n=64, Q_stage=2, dO_stage=1, PdS_stage=2, SdP_swapAB=False, AtomLayoutNdKV=1),
}

# Plain in-process caches: FA4's disk cache fingerprints only flash_attn/cute sources, so edits
# to our kernels would load stale binaries.
_JVP_FWD_CACHE = {}
_HVP_BWD_CACHE = {}


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def _size1_flags(tensors):
    return tuple(tuple(s == 1 for s in t.shape) for t in tensors)


def _compile_and_run(cache, kernel_name, make_obj, key_extra, tensor_args, softmax_scale, cu_seqlens_q, cu_seqlens_k):
    key = (
        kernel_name,
        *key_extra,
        cu_seqlens_q is not None,
        cu_seqlens_k is not None,
        *_size1_flags(tensor_args),
    )
    if key not in cache:
        cache[key] = cute.compile(
            make_obj(),
            *[to_cute_tensor(t) for t in tensor_args],
            softmax_scale,
            *[
                to_cute_tensor(t, assumed_align=4) if t is not None else None
                for t in (cu_seqlens_q, cu_seqlens_k)
            ],
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    cache[key](*tensor_args, softmax_scale, cu_seqlens_q, cu_seqlens_k)


def _jvp_fwd(
    q, k, v, gdq, gdk, gdv, out, dout, lse_log2, dot_dout, z1, ddot,
    softmax_scale, causal, cu_seqlens_q, cu_seqlens_k,
):
    """K1: dot_dout (input dtype); z1 and ddot (fp32, padded like lse_log2)."""
    cute_dtype = torch2cute_dtype_map[q.dtype]
    head_dim = q.shape[-1]
    qhead_per_kvhead = q.shape[-2] // k.shape[-2]
    _compile_and_run(
        _JVP_FWD_CACHE,
        "jvp_fwd",
        lambda: FlashbackJvpFwdSm90(
            cute_dtype,
            head_dim,
            qhead_per_kvhead,
            causal,
            tile_m_bwd=K2_TILE_M,
            tile_m=K1_TILE_M,
            tile_n=K1_TILE_N,
            num_stages=K1_NUM_STAGES,
        ),
        (cute_dtype, head_dim, qhead_per_kvhead, causal),
        (q, k, v, gdq, gdk, gdv, out, dout, lse_log2, dot_dout, z1, ddot),
        softmax_scale,
        cu_seqlens_q,
        cu_seqlens_k,
    )


def _hvp_bwd(
    q, k, v, dout, gdq, gdk, gdv, lse_log2, dpsum, z1, ddot, dotq_accum, dot_k, dot_v,
    softmax_scale, causal, cu_seqlens_q, cu_seqlens_k,
):
    """K2: dotq_accum (fp32, unscaled), dot_k/dot_v (input dtype for MHA, fp32 accum for GQA)."""
    cute_dtype = torch2cute_dtype_map[q.dtype]
    head_dim = q.shape[-1]
    qhead_per_kvhead = q.shape[-2] // k.shape[-2]
    _compile_and_run(
        _HVP_BWD_CACHE,
        "hvp_bwd",
        lambda: FlashbackHvpBwdSm90(
            cute_dtype,
            head_dim,
            qhead_per_kvhead,
            causal,
            tile_m=K2_TILE_M,
            **K2_CONFIGS[head_dim],
        ),
        (cute_dtype, head_dim, qhead_per_kvhead, causal),
        (q, k, v, dout, gdq, gdk, gdv, lse_log2, dpsum, z1, ddot, dotq_accum, dot_k, dot_v),
        softmax_scale,
        cu_seqlens_q,
        cu_seqlens_k,
    )


def _double_backward(
    q, k, v, out, dout, lse, gdq, gdk, gdv,
    softmax_scale, causal, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
):
    """Returns (dot_q, dot_k, dot_v, dot_dout): grads of <(dq, dk, dv), (gdq, gdk, gdv)>
    w.r.t. (q, k, v, dout)."""
    if q.numel() == 0 or k.numel() == 0:
        return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v), torch.zeros_like(dout)
    q, k, v, out, dout, gdq, gdk, gdv = [
        t if t.is_contiguous() else t.contiguous() for t in (q, k, v, out, dout, gdq, gdk, gdv)
    ]
    device = q.device
    cute_dtype = torch2cute_dtype_map[q.dtype]
    head_dim = q.shape[-1]
    num_head, num_head_kv = q.shape[-2], k.shape[-2]
    f32 = dict(dtype=torch.float32, device=device)

    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        seqlen_q_rounded = _round_up(seqlen_q, K2_TILE_M)
        stats_shape = (batch_size, num_head, seqlen_q_rounded)
        dotq_accum = torch.empty(batch_size, num_head, seqlen_q_rounded * head_dim, **f32)
    else:
        total_q = q.shape[0]
        total_q_rounded_padded = (total_q + cu_seqlens_q.shape[0] * K2_TILE_M - 1) // K2_TILE_M * K2_TILE_M
        stats_shape = (num_head, total_q_rounded_padded)
        dotq_accum = torch.empty(num_head, total_q_rounded_padded * head_dim, **f32)
    dpsum = torch.empty(stats_shape, **f32)
    lse_log2 = torch.empty(stats_shape, **f32)
    # K2 reads z1 / ddot on padding / gap rows (multiplied by P = 0) that K1 never writes: they
    # must not hold NaN garbage. One allocation keeps it a single memset.
    z1, ddot = torch.zeros((2, *stats_shape), **f32).unbind(0)

    dot_dout = torch.empty_like(dout)
    dot_q = torch.empty_like(q)
    dot_k = torch.empty_like(k)
    dot_v = torch.empty_like(v)
    is_gqa = num_head > num_head_kv
    if is_gqa:
        k2_tile_n = K2_CONFIGS[head_dim]["tile_n"]
        if cu_seqlens_k is None:
            batch_size, seqlen_k = k.shape[:2]
            seqlen_k_rounded = _round_up(seqlen_k, k2_tile_n)
            accum_shape = (batch_size, num_head_kv, seqlen_k_rounded * head_dim)
        else:
            total_k = k.shape[0]
            total_k_rounded_padded = (total_k + cu_seqlens_k.shape[0] * k2_tile_n - 1) // k2_tile_n * k2_tile_n
            accum_shape = (num_head_kv, total_k_rounded_padded * head_dim)
        dotk_accum = torch.zeros(accum_shape, **f32)
        dotv_accum = torch.zeros(accum_shape, **f32)

    # D = rowsum(dO * O), lse_log2, zeroed dotq_accum.
    _bwd_preprocess(
        out, dout, dpsum, lse, lse_log2, dotq_accum,
        cu_seqlens_q, None, None,
        cute_dtype, head_dim, head_dim, K2_TILE_M,
        fake_mode=False,
        hdim_multiple_of=32,
    )
    _jvp_fwd(
        q, k, v, gdq, gdk, gdv, out, dout, lse_log2, dot_dout, z1, ddot,
        softmax_scale, causal, cu_seqlens_q, cu_seqlens_k,
    )
    _hvp_bwd(
        q, k, v, dout, gdq, gdk, gdv, lse_log2, dpsum, z1, ddot, dotq_accum,
        dotk_accum if is_gqa else dot_k,
        dotv_accum if is_gqa else dot_v,
        softmax_scale, causal, cu_seqlens_q, cu_seqlens_k,
    )
    num_threads_post = 256  # 2 MMA warpgroups
    _bwd_postprocess_convert(
        dotq_accum, dot_q, softmax_scale,
        cu_seqlens_q, None,
        90, cute_dtype, head_dim, K2_TILE_M, num_threads_post,
        1, False,
        use_2cta_instrs=False, cluster_size=1,
        fake_mode=False,
        hdim_multiple_of=32,
    )
    if is_gqa:
        # Layout of the fp32 dK/dV accumulators written by K2's GQA epilogue.
        dkv_layout = (k2_tile_n, num_threads_post, K2_CONFIGS[head_dim]["AtomLayoutNdKV"], False)
        for accum, dst, scale in ((dotk_accum, dot_k, softmax_scale), (dotv_accum, dot_v, 1.0)):
            _bwd_postprocess_convert(
                accum, dst, scale,
                cu_seqlens_k, None,
                90, cute_dtype, head_dim, *dkv_layout,
                cluster_size=1,
                fake_mode=False,
                hdim_multiple_of=32,
            )
    return dot_q, dot_k, dot_v, dot_dout


class _NoThirdOrderGrad(torch.autograd.Function):
    """Identity on the double-backward results whose backward raises.

    `once_differentiable` only guards when the incoming grad_outputs require grad, so results
    depending on q/k/v/dout through saved tensors would silently be treated as constants.
    """

    @staticmethod
    def forward(ctx, num_outputs, *tensors):
        return tuple(t.view_as(t) for t in tensors[:num_outputs])

    @staticmethod
    def backward(ctx, *grads):
        raise RuntimeError(
            "flashback_cute supports derivatives up to second order; "
            "the double-backward results cannot be differentiated again"
        )


class _FlashbackAttnBwdFunc(torch.autograd.Function):
    """(q, k, v, dout) -> (dq, dk, dv) via FA4's backward; its backward is the fused double backward.

    out and lse enter as constants: the returned grads are total derivatives w.r.t. q, k, v, dout,
    already including the dependence through O and LSE.
    """

    @staticmethod
    def forward(ctx, q, k, v, dout, out, lse, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal):
        dq, dk, dv = _flash_attn_bwd(
            q, k, v, out, dout, lse, softmax_scale, causal,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )
        ctx.save_for_backward(q, k, v, dout, out, lse, cu_seqlens_q, cu_seqlens_k)
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return dq, dk, dv

    @staticmethod
    def backward(ctx, gdq, gdk, gdv):
        q, k, v, dout, out, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        with torch.no_grad():
            grads = _double_backward(
                q, k, v, out, dout, lse, gdq, gdk, gdv,
                ctx.softmax_scale, ctx.causal, cu_seqlens_q, cu_seqlens_k, ctx.max_seqlen_q, ctx.max_seqlen_k,
            )
        deps = (q, k, v, dout, gdq, gdk, gdv)
        if torch.is_grad_enabled() and any(t.requires_grad for t in deps):
            grads = _NoThirdOrderGrad.apply(len(grads), *grads, *deps)
        return *grads, *((None,) * 8)


class _FlashbackAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal):
        out, lse, _, _ = _flash_attn_fwd(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            return_lse=True,
        )
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        dq, dk, dv = _FlashbackAttnBwdFunc.apply(
            q, k, v, dout, out.detach(), lse.detach(),
            cu_seqlens_q, cu_seqlens_k, ctx.max_seqlen_q, ctx.max_seqlen_k,
            ctx.softmax_scale, ctx.causal,
        )
        return dq, dk, dv, *((None,) * 6)


def _validate(q, k, v):
    if torch.cuda.get_device_capability(q.device)[0] != 9:
        raise NotImplementedError("flashback_cute supports only SM90 (Hopper) GPUs")
    if not (q.dtype == k.dtype == v.dtype) or q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("q, k, v must share a dtype in {float16, bfloat16}")
    if not (k.shape[-1] == v.shape[-1] == q.shape[-1]) or q.shape[-1] not in (64, 128):
        raise NotImplementedError("head_dim must equal head_dim_v and be 64 or 128")
    if q.shape[-2] % k.shape[-2] != 0:
        raise ValueError("number of q heads must be divisible by number of kv heads")


def flashback_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """Softmax attention, q (b, sq, hq, d), k/v (b, sk, hkv, d); twice differentiable.

    Causal masking is bottom-right aligned (FA4 convention).
    """
    _validate(q, k, v)
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    return _FlashbackAttnFunc.apply(q, k, v, None, None, None, None, softmax_scale, causal)


def flashback_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """Varlen softmax attention, q (total_q, hq, d), k/v (total_k, hkv, d), cu_seqlens int32 (b+1,)."""
    _validate(q, k, v)
    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
        raise TypeError("cu_seqlens_q and cu_seqlens_k must be int32")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    return _FlashbackAttnFunc.apply(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal
    )
