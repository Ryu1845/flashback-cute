"""Pure-PyTorch attention reference used as the test oracle and benchmark baseline."""

import torch


def attention_ref(q, k, v, causal: bool, softmax_scale: float, upcast: bool):
    """Softmax attention, q (b, sq, hq, d), k/v (b, sk, hkv, d) -> (b, sq, hq, d).

    upcast=True: everything in float64. upcast=False: matmuls in the input dtype,
    softmax internals in float32, P cast back to the input dtype before P @ V.
    Causal masking is bottom-right aligned (FA4 convention). Fully masked rows give 0.
    """
    dtype = q.dtype
    if upcast:
        q, k, v = q.double(), k.double(), v.double()
    hq, hkv = q.shape[2], k.shape[2]
    if hq != hkv:
        k = k.repeat_interleave(hq // hkv, dim=2)
        v = v.repeat_interleave(hq // hkv, dim=2)
    sq, sk = q.shape[1], k.shape[1]
    s = softmax_scale * torch.einsum("bqhd,bkhd->bhqk", q, k)
    if not upcast:
        s = s.float()
    if causal:
        row = torch.arange(sq, device=q.device)[:, None]
        col = torch.arange(sk, device=q.device)[None, :]
        mask = col <= row + (sk - sq)
        s = s.masked_fill(~mask, float("-inf"))
    m = s.amax(-1, keepdim=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m)).detach()
    e = torch.exp(s - m)
    p = e / e.sum(-1, keepdim=True).clamp_min(1e-300 if upcast else 1e-30)
    if not upcast:
        p = p.to(dtype)
    return torch.einsum("bhqk,bkhd->bqhd", p, v)


def attention_ref_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, causal: bool, softmax_scale: float, upcast: bool):
    """Varlen attention: q (total_q, hq, d), k/v (total_k, hkv, d), cu_seqlens int32 (b+1,)."""
    cq = cu_seqlens_q.tolist()
    ck = cu_seqlens_k.tolist()
    outs = []
    for b in range(len(cq) - 1):
        outs.append(
            attention_ref(
                q[None, cq[b] : cq[b + 1]],
                k[None, ck[b] : ck[b + 1]],
                v[None, ck[b] : ck[b + 1]],
                causal,
                softmax_scale,
                upcast,
            )[0]
        )
    return torch.cat(outs, dim=0)


def double_backward(fn, q, k, v, dout, gq, gk, gv):
    """Returns (out, dq, dk, dv, dot_q, dot_k, dot_v, dot_dout).

    q, k, v, dout must be leaves with requires_grad=True; (gq, gk, gv) are the cotangents
    of (dq, dk, dv) for the second backward.
    """
    out = fn(q, k, v)
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout, create_graph=True)
    dot = torch.autograd.grad((dq, dk, dv), (q, k, v, dout), (gq, gk, gv))
    return (out, dq, dk, dv, *dot)
