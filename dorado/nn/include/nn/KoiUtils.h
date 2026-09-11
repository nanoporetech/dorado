#pragma once

#include <cuda_runtime_api.h>

namespace dorado::nn {

// TODO: These should really be part of Koi
bool koi_can_use_cutlass(/* current device */);
bool koi_can_use_cutlass(int device_id);
bool koi_can_use_quantised_lstm(/* current device */);
bool koi_can_run_flstm(/* current device */);
bool koi_can_run_flstm(int device_id);
bool koi_can_run_tx_vcs(int device_id);

// NVFP4 tx_encoder
std::pair<at::Tensor, at::Tensor> to_block_fp4(const at::Tensor& t_in, const float maxval);
at::Tensor convert_to_nvfp4_tiled(const at::Tensor& t,
                                  const std::array<int64_t, 2>& smem_dims,
                                  const bool is_B,
                                  const at::TensorOptions i32_opts,
                                  const cudaStream_t stream_ptr);
at::Tensor untile_nvfp4(const at::Tensor& t_nvfp4,
                        const std::array<int64_t, 2>& smem_dims,
                        const at::TensorOptions f16_opts,
                        const cudaStream_t stream_ptr);

}  // namespace dorado::nn
