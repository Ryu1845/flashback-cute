import itertools

import pytest
import torch

from flashback_cute import flashback_attn_func, flashback_attn_varlen_func
from tests.reference import attention_ref, attention_ref_varlen, double_backward

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
    pytest.skip("flashback_cute requires an SM90 GPU", allow_module_level=True)

OUTPUT_NAMES = ("out", "dq", "dk", "dv", "dot_q", "dot_k", "dot_v", "dot_dout")


def _check_against_refs(kernel_fn, ref_fn, q, k, v):
    """Every output of (forward, backward, double backward) must be within 3x the error of a
    same-dtype PyTorch reference, both measured against an fp64 reference."""
    dout = torch.randn_like(q)
    gq, gk, gv = torch.randn_like(q), torch.randn_like(k), torch.randn_like(v)
    inputs = (q, k, v, dout)

    res = double_backward(kernel_fn, *[t.detach().clone().requires_grad_() for t in inputs], gq, gk, gv)
    ref = double_backward(
        lambda q_, k_, v_: ref_fn(q_, k_, v_, upcast=True),
        *[t.detach().double().requires_grad_() for t in inputs],
        gq.double(),
        gk.double(),
        gv.double(),
    )
    pt = double_backward(
        lambda q_, k_, v_: ref_fn(q_, k_, v_, upcast=False),
        *[t.detach().clone().requires_grad_() for t in inputs],
        gq,
        gk,
        gv,
    )
    for name, x, r, p in zip(OUTPUT_NAMES, res, ref, pt):
        assert x.shape == r.shape, name
        err = (x.double() - r).abs().max().item()
        err_pt = (p.double() - r).abs().max().item()
        tol = 3 * err_pt + 1e-3 * r.abs().max().item()
        assert err <= tol, f"{name}: max err {err:.3e} > tol {tol:.3e} (pt err {err_pt:.3e})"


def _check_dense(dtype, d, hq, hkv, causal, sq, sk, batch=2):
    torch.manual_seed(0)
    q = torch.randn(batch, sq, hq, d, device="cuda", dtype=dtype)
    k = torch.randn(batch, sk, hkv, d, device="cuda", dtype=dtype)
    v = torch.randn(batch, sk, hkv, d, device="cuda", dtype=dtype)
    scale = d**-0.5
    _check_against_refs(
        lambda q, k, v: flashback_attn_func(q, k, v, softmax_scale=scale, causal=causal),
        lambda q, k, v, upcast: attention_ref(q, k, v, causal, scale, upcast),
        q,
        k,
        v,
    )


def _check_varlen(dtype, d, hq, hkv, causal, lens_q, lens_k):
    torch.manual_seed(0)
    cu_q = torch.tensor([0, *itertools.accumulate(lens_q)], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, *itertools.accumulate(lens_k)], device="cuda", dtype=torch.int32)
    q = torch.randn(sum(lens_q), hq, d, device="cuda", dtype=dtype)
    k = torch.randn(sum(lens_k), hkv, d, device="cuda", dtype=dtype)
    v = torch.randn(sum(lens_k), hkv, d, device="cuda", dtype=dtype)
    scale = d**-0.5
    _check_against_refs(
        lambda q, k, v: flashback_attn_varlen_func(
            q, k, v, cu_q, cu_k, max(lens_q), max(lens_k), softmax_scale=scale, causal=causal
        ),
        lambda q, k, v, upcast: attention_ref_varlen(q, k, v, cu_q, cu_k, causal, scale, upcast),
        q,
        k,
        v,
    )


@pytest.mark.parametrize("seqlens", [(128, 128), (113, 203), (256, 97)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("heads", [(4, 4), (6, 2)])
@pytest.mark.parametrize("d", [64, 128])
def test_dense_bf16(d, heads, causal, seqlens):
    _check_dense(torch.bfloat16, d, *heads, causal, *seqlens)


@pytest.mark.parametrize("causal", [False, True])
def test_dense_mqa(causal):
    _check_dense(torch.bfloat16, 64, 4, 1, causal, 200, 200)


@pytest.mark.parametrize("d", [64, 128])
def test_dense_fp16(d):
    _check_dense(torch.float16, d, 4, 4, True, 200, 200)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("heads", [(4, 4), (4, 2)])
@pytest.mark.parametrize("d", [64, 128])
def test_varlen_bf16(d, heads, causal):
    _check_varlen(torch.bfloat16, d, *heads, causal, [37, 200, 1, 129], [50, 200, 7, 300])
