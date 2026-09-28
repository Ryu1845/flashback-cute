"""Benchmark flashback_cute vs. naive PyTorch attention on the flashback pipelines.

  fwd             out = f(q, k, v)                                   (no autograd tape)
  fwd+bwd         grad(f(q, k, v).sum(), (q, k, v))
  fwd+bwd+bwdbwd  bck = grad(f(q, k, v).sum(), (q, k, v), create_graph=True)
                  grad(sum(t.sum() for t in bck), (q, k, v))

Run from the repo root: uv run python bench/bench_double_backward.py [--causal]
"""

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from flashback_cute import flashback_attn_func  # noqa: E402
from tests.reference import attention_ref  # noqa: E402

PIPELINES = ("fwd", "fwd+bwd", "fwd+bwd+bwdbwd")
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def run_pipeline(name, fn, q, k, v):
    if name == "fwd":
        with torch.no_grad():
            fn(q, k, v)
    elif name == "fwd+bwd":
        torch.autograd.grad(fn(q, k, v).sum(), (q, k, v))
    else:
        bck = torch.autograd.grad(fn(q, k, v).sum(), (q, k, v), create_graph=True)
        torch.autograd.grad(sum(t.sum() for t in bck), (q, k, v))


def measure(name, fn, q, k, v, warmup=3, iters=10):
    """Returns (median ms, peak allocated GB) or None on OOM."""
    try:
        for _ in range(warmup):
            run_pipeline(name, fn, q, k, v)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run_pipeline(name, fn, q, k, v)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        return statistics.median(times), torch.cuda.max_memory_allocated() / 2**30
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--hdim", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--seqlens", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--causal", action="store_true")
    args = parser.parse_args()
    dtype = DTYPES[args.dtype]

    print(
        f"batch={args.batch} heads={args.heads} dtype={args.dtype} causal={args.causal} "
        f"({torch.cuda.get_device_name()})"
    )
    header = f"{'method':<15}{'hdim':>5}{'seqlen':>7}" + "".join(f"{p + ' ms':>21}{'GB':>7}" for p in PIPELINES)
    print(header)
    print("-" * len(header))
    for hdim in args.hdim:
        scale = hdim**-0.5
        methods = {
            "flashback_cute": lambda q, k, v: flashback_attn_func(q, k, v, softmax_scale=scale, causal=args.causal),
            "naive": lambda q, k, v: attention_ref(q, k, v, args.causal, scale, upcast=False),
        }
        for method, fn in methods.items():
            for seqlen in args.seqlens:
                torch.manual_seed(0)
                q, k, v = [
                    torch.randn(args.batch, seqlen, args.heads, hdim, device="cuda", dtype=dtype, requires_grad=True)
                    for _ in range(3)
                ]
                row = f"{method:<15}{hdim:>5}{seqlen:>7}"
                for pipeline in PIPELINES:
                    res = measure(pipeline, fn, q, k, v)
                    row += f"{'OOM':>21}{'-':>7}" if res is None else f"{res[0]:>21.3f}{res[1]:>7.2f}"
                print(row, flush=True)
                del q, k, v
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
