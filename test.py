# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import argparse
import math
import os

import torch

from torch.profiler import profile
from triton.testing import do_bench  # @manual

from .triton_kernel import matmul as triton_matmul


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    try:
        torch.ops.load_library("cutlass.so")
    except Exception:
        torch.ops.load_library("//scripts/bertrand/tf32_gemm:cutlass")

    m = 9 * 128
    n = 12 * 256
    k = 4096

    a = torch.randn(m, k, device="cuda").div(math.sqrt(k))
    b = torch.randn(k, n, device="cuda").div(math.sqrt(k))

    c_cutlass = torch.ops.cutlass.gemm(a, b)
    c_triton = triton_matmul(a, b)

    c_ref = a @ b

    print(c_ref)
    print(c_cutlass)
    print(c_triton)

    print("triton allclose:", torch.allclose(c_triton, c_ref, atol=1e-4, rtol=1e-4))
    print(
        "triton (precompiled) allclose:",
        torch.allclose(
            c_triton, triton_matmul(a, b, precompiled=True), atol=1e-4, rtol=1e-4
        ),
    )
    print("cutlass allclose:", torch.allclose(c_cutlass, c_ref, atol=1e-4, rtol=1e-4))

    tflops = 2 * m * n * k / 1e9

    print("torch (cublas):", tflops / do_bench(lambda: torch.mm(a, b)))
    print("cutlass:", tflops / do_bench(lambda: torch.ops.cutlass.gemm(a, b)))
    print("triton:", tflops / do_bench(lambda: triton_matmul(a, b)))
    print(
        "triton (precompiled):",
        tflops / do_bench(lambda: triton_matmul(a, b, precompiled=True)),
    )

    if args.profile:
        with profile() as p:
            torch.zeros(1)
            for _ in range(3):
                torch.mm(a, b)
            torch.cuda.synchronize()
            for _ in range(3):
                torch.ops.cutlass.gemm(a, b)
            torch.cuda.synchronize()
            for _ in range(3):
                triton_matmul(a, b)
            torch.cuda.synchronize()
        p.export_chrome_trace("gemm.json.gz")
        os.system(
            "manifold put --overwrite --threads 20 gemm.json.gz gpu_traces/tree/traces/bertrand/gemm.json.gz"
        )
        print(
            "https://www.internalfb.com/intern/perfdoctor/trace_view?filepath=tree/traces/bertrand/gemm.json.gz&bucket=gpu_traces"
        )


if __name__ == "__main__":
    main()
