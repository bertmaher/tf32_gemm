// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal.h"

/**
 * Panic wrapper for unwinding CUTLASS errors
 */
#define CUTLASS_CHECK(status)                                             \
  {                                                                       \
    cutlass::Status error = status;                                       \
    if (error != cutlass::Status::kSuccess) {                             \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) \
                << " at: " << __LINE__ << std::endl;                      \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  }

///////////////////////////////////////////////////////////////////////////////////////////////////

// The code section below describes datatype for input, output matrices and
// computation between elements in input matrices.
using ElementAccumulator = float; // <- data type of accumulator
using ElementComputeEpilogue =
    ElementAccumulator; // <- data type of epilogue operations
using ElementInputA = float; // <- data type of elements in input matrix A
using ElementInputB = float; // <- data type of elements in input matrix B
using ElementOutput = float; // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices.
// Column Major for Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular
// SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 256, 16>; // <- threadblock tile M = 128, N =
                                            // 128, K = 16
// This code section describes tile size a warp will compute
using ShapeMMAWarp =
    cutlass::gemm::GemmShape<64, 64, 16>; // <- warp tile M = 64, N = 64, K = 16
// This code section describes the size of MMA op
using ShapeMMAOp =
    cutlass::gemm::GemmShape<16, 8, 8>; // <- MMA Op tile M = 16, N = 8, K = 8

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock =
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // <- ??

// This code section describes the epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, // <- data type of output matrix
    128 /
        cutlass::sizeof_bits<
            ElementOutput>::value, // <- the number of elements per vectorized
                                   // memory access. For a byte, it's 16
                                   // elements. This becomes the vector width of
                                   // math instructions in the epilogue too
    ElementAccumulator, // <- data type of accumulator
    ElementComputeEpilogue>; // <- data type for alpha/beta in linear
                             // combination function

// Number of pipelines you want to use
constexpr int NumStages = 3;

using Gemm = cutlass::gemm::device::Gemm<
    ElementInputA,
    LayoutInputA,
    ElementInputB,
    LayoutInputB,
    ElementOutput,
    LayoutOutput,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ShapeMMAThreadBlock,
    ShapeMMAWarp,
    ShapeMMAOp,
    EpilogueOp,
    SwizzleThreadBlock,
    NumStages>;

void gemm_kernel(float* a, float* b, float* c, int m, int n, int k) {
  cutlass::gemm::GemmCoord problem_size{m, n, k};
  cutlass::TensorRef tensor_a{a, LayoutInputA{k}};
  cutlass::TensorRef tensor_b{b, LayoutInputB{n}};
  cutlass::TensorRef tensor_c{c, LayoutOutput{n}};
  cutlass::TensorRef tensor_d{c, LayoutOutput{n}};

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1.0f);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0.0f);

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments
  // to launch instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{
      problem_size, // <- problem size of matrix multiplication
      tensor_a, // <- reference to matrix A on device
      tensor_b, // <- reference to matrix B on device
      tensor_c, // <- reference to matrix C on device
      tensor_d, // <- reference to matrix D on device
      {alpha, beta}, // <- tuple of alpha and beta
      split_k_slices}; // <- k-dimension split factor

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // printf("workspace size: %d\n", workspace_size);
  if (workspace_size != 0) {
    exit(EXIT_FAILURE);
  }
  // Allocate workspace memory
  // cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  Gemm gemm_op;

  // Instantiate CUTLASS kernel depending on templates
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  status = gemm_op.initialize(arguments, nullptr); // workspace.get());
  CUTLASS_CHECK(status);

  status = gemm_op();
  CUTLASS_CHECK(status);
}
