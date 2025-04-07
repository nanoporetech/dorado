#pragma once

#if DORADO_CUDA_BUILD

namespace dorado::nn {

// We have three different LSTM code paths:
//
// - Quantized: This path is only available for narrow LSTM layers, C == 96 or C == 128. It
//   uses CuBLAS GEMM (or torch::matmul) for the FP16 input-hidden matmul, and a custom kernel
//   using the DP4A instruction for the Int8*Int8->Int32 hidden-hidden matmul, and FP16 gate
//   computation. DP4A is not available on compute arch 6.2 (TX2).
//
// - Cutlass: This path is only available for LSTM layers where C is a multiple of 128 between
//   256 and 1024. It is currently only available on compute arch 8.0 (A100) and 9.0 (H100).
//   It uses a custom kernel based on the Cutlass library which performs Tensor Core matmul using
//   either F16 or Int8 and fuses the gate computation. FP16 is used only for the first LSTM layer,
//   and only if the output activation of the last convolution is not tanh.
// TODO: Add Cutlass kernels for 7.0 (V100, FP16) and for GPUs with less shared memory (7.x, 8.x)
//
// - CuBLAS: Slowest. This is the fallback path when none of the other paths applies. It uses
//   CuBLAS GEMM (or torch::matmul) plus `host_lstm_step_f16` from Koi. Uses FP16 precision.
//
// Each path needs its input in a different memory layout. To avoid extra transpose/conversion
// steps, the last convolution writes output in a memory layout suitable to serve as working memory
// for the first LSTM layer. (The specific memory layouts are further explained below in
// `LSTMStackImpl::forward_[cublas|cutlass]`.)
//
// These are the possible memory layouts for in/out buffers in working memory:
// [where T = chunk size ("time"), N = batch size, C = layer size ("channels")]
//
// - NTC: A contiguous tensor of size [N, T, C], dtype torch::kF16
// - TNC: A contiguous tensor of size [T, N, C], dtype torch::kF16
// - CUTLASS_TNC_F16: a contiguous tensor of size [T + 3, N, C], dtype torch::kF16
// - CUTLASS_TNC_I8: a contiguous tensor of size [T + 3, N, C], dtype torch::kI8
// - CUBLAS_TN2C: a contiguous tensor of size [T + 1, N, 2, C], dtype torch::kF16
//

// TODO: These should really be part of Koi
bool koi_can_use_cutlass();
bool koi_can_use_quantised_lstm();

}  // namespace dorado::nn

#endif