// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "cutlass_kernel.h"

#include "ATen/ATen.h" // @manual
#include "torch/extension.h" // @manual

at::Tensor gemm(at::Tensor a, at::Tensor b) {
  auto c = a.new_empty({a.size(0), b.size(1)});
  gemm_kernel(
      a.data_ptr<float>(),
      b.data_ptr<float>(),
      c.data_ptr<float>(),
      a.size(0),
      b.size(1),
      a.size(1));
  return c;
}

TORCH_LIBRARY(cutlass, m) {
  m.def("gemm", &gemm);
}
