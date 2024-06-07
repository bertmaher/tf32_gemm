# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch
import triton  # @manual

from triton_kernel import matmul as triton_matmul

try:
    torch.ops.load_library("cutlass.so")
except Exception:
    torch.ops.load_library("//scripts/bertrand/tf32_gemm:cutlass")

torch.set_float32_matmul_precision("high")

configs = []
for fp8_inputs in [False]:
    configs.append(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
            x_vals=[128 * i for i in range(2, 33)],
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
            line_vals=["cublas", "triton", "cutlass", "precompiled"],
            line_names=["cublas", "triton", "cutlass", "precompiled"],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="matmul-performance-fp32",
            args={"fp8_inputs": fp8_inputs},
        )
    )


@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider, fp8_inputs):
    a = torch.zeros((M, K), device="cuda", dtype=torch.float32)
    b = torch.zeros((K, N), device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "cublas":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.matmul(a, b), quantiles=quantiles
        )
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_matmul(a, b), quantiles=quantiles
        )
        # print(f"{N}: {matmul_kernel.best_config}")
    if provider == "precompiled":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_matmul(a, b, precompiled=True), quantiles=quantiles
        )
        # print(f"{N}: {matmul_kernel.best_config}")
    if provider == "cutlass":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.ops.cutlass.gemm(a, b), quantiles=quantiles
        )

    def perf(ms):
        return 2 * M * N * K * 1e-12 / (ms * 1e-3)

    return perf(ms), perf(max_ms), perf(min_ms)


def main():
    benchmark.run(show_plots=True, print_data=True, save_path=".")


if __name__ == "__main__":
    main()
