#include "nn/KoiUtils.h"

#include <ATen/cuda/CUDAContextLight.h>

namespace {

bool koi_can_use_cutlass_internal(const cudaDeviceProp* const prop) { return (prop->major >= 8); }

bool koi_can_run_flstm_internal(const cudaDeviceProp* const prop) {
    return (prop->major >= 8) && (prop->multiProcessorCount >= 9);
}

}  // namespace

namespace dorado::nn {

// TODO: These should really be part of Koi

bool koi_can_use_cutlass() {
    return koi_can_use_cutlass_internal(at::cuda::getCurrentDeviceProperties());
}
bool koi_can_use_cutlass(const int device_id) {
    return koi_can_use_cutlass_internal(at::cuda::getDeviceProperties(device_id));
}

bool koi_can_use_quantised_lstm() {
    cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
    // DP4A is supported on Pascal and later, except for TX2 (sm_62).
    return (prop->major > 6) || (prop->major == 6 && prop->minor != 2);
}

bool koi_can_run_flstm() {
    return koi_can_run_flstm_internal(at::cuda::getCurrentDeviceProperties());
}
bool koi_can_run_flstm(const int device_id) {
    return koi_can_run_flstm_internal(at::cuda::getDeviceProperties(device_id));
}
bool koi_can_run_tx_vcs(const int device_id) {
    cudaDeviceProp* prop = at::cuda::getDeviceProperties(device_id);
    return (prop->major >= 8);
}

#if DORADO_CUDA_BUILD
extern "C" {
#include "koi.h"
}

std::pair<at::Tensor, at::Tensor> to_block_fp4(const at::Tensor& t_in, const float maxval) {
    TORCH_CHECK(t_in.dim() == 3, "to_block_fp4 expects 3 dimensions, [M_or_N, K // 16, 16], got ",
                t_in.dim(), " dimensions");
    TORCH_CHECK(t_in.size(-1) == 16, "to_block_fp4 expects last dim == 16");

    auto si = t_in.sign();
    auto t = t_in.abs().to(c10::kFloat);
    auto tmax = t.amax(/*dim=*/-1);

    auto scale_f = tmax / maxval;
    auto t_in_sf = scale_f.to(c10::kFloat8_e4m3fn);

    auto scale_float = t_in_sf.to(c10::kFloat).unsqueeze(-1);
    auto x = t / scale_float;
    auto t_in_scaled = x.mul(si);  // si is promoted to fp32

    return {t_in_scaled.reshape({t_in.size(0), t_in.size(1) * 16}).to(c10::kHalf), t_in_sf};
}

at::Tensor convert_to_nvfp4_tiled(const at::Tensor& t,
                                  const std::array<int64_t, 2>& smem_dims,
                                  const bool is_B,
                                  const at::TensorOptions i32_opts,
                                  const cudaStream_t stream_ptr) {
    TORCH_CHECK(t.dim() == 2, "convert_to_nvfp4_tiled expects 2 dimensions, [M_or_N, K], got ",
                t.dim(), " dimensions");
    // Tensors must always be K-major
    const int64_t dim0 = t.size(0);
    const int64_t dim1 = t.size(1);
    const int64_t smem_dim0 = smem_dims[0];
    const int64_t smem_dim1 = smem_dims[1];

    auto [t_scaled, t_sf] = to_block_fp4(t.view({dim0, dim1 / 16, 16}), 6.f);

    std::vector<int64_t> fp4_shape = {dim0 / smem_dim0, smem_dim0 / 32, 2, 2, 8,
                                      dim1 / smem_dim1, smem_dim1 / 64, 2, 4, 8};
    std::vector<int64_t> fp4_permute = is_B ? std::vector<int64_t>{0, 5, 1, 6, 2, 4, 8, 3, 7, 9}
                                            : std::vector<int64_t>{0, 5, 1, 6, 2, 4, 8, 7, 3, 9};
    t_scaled = t_scaled.view(fp4_shape).permute(fp4_permute).contiguous();
    at::Tensor t_fp4 = at::empty(
            {dim0 / smem_dim0, dim1 / smem_dim1, (smem_dim0 / 32) * (smem_dim1 / 64), 256},
            i32_opts);
    koi_convert_fp16_to_fp4(stream_ptr, t_fp4.data_ptr(), t_scaled.data_ptr(), dim0 * dim1);

    std::vector<int64_t> sf_shape = {dim0 / smem_dim0, smem_dim0 / 32, 4, 8,
                                     dim1 / smem_dim1, smem_dim1 / 64, 4};
    std::vector<int64_t> sf_permute = std::vector<int64_t>{0, 4, 1, 5, 3, 2, 6};
    t_sf = t_sf.view(sf_shape).permute(sf_permute).contiguous();
    auto t_sf_fp8x4 = t_sf.view(c10::kInt);
    t_sf_fp8x4 = t_sf_fp8x4
                         .view({dim0 / smem_dim0, dim1 / smem_dim1,
                                (smem_dim0 / 32) * (smem_dim1 / 64), 32})
                         .contiguous();

    at::Tensor t_nvfp4 = at::cat({t_fp4, t_sf_fp8x4}, -1);
    return t_nvfp4;
}

at::Tensor untile_nvfp4(const at::Tensor& t_nvfp4,
                        const std::array<int64_t, 2>& smem_dims,
                        const at::TensorOptions f16_opts,
                        const cudaStream_t stream_ptr) {
    TORCH_CHECK(t_nvfp4.scalar_type() == c10::kInt);
    TORCH_CHECK(t_nvfp4.dim() == 4, "A koi NVFP4 tensor needs to have 4 dimensions: ",
                "[M / SMEM_M, N / SMEM_N, (SMEM_M / 32) * (SMEM_N / 64), 288]");

    const int64_t SMEM_M = smem_dims[0];
    const int64_t SMEM_N = smem_dims[1];
    const int64_t M = t_nvfp4.size(0) * SMEM_M;
    const int64_t N = t_nvfp4.size(1) * SMEM_N;

    at::Tensor t_fp4 = t_nvfp4.slice(3, 0, 256);
    at::Tensor t_fp4_sim = at::empty({M * N}, f16_opts);
    koi_convert_fp4_to_fp16(stream_ptr, t_fp4.data_ptr(), t_fp4_sim.data_ptr(), M * N);
    at::Tensor t_sf = t_nvfp4.slice(3, 256, 288).view(c10::kFloat8_e4m3fn);

    t_fp4_sim = t_fp4_sim.view({M / SMEM_M, N / SMEM_N, SMEM_M / 32, SMEM_N / 64, 2, 8, 4, 2, 2, 8})
                        .permute({0, 2, 4, 8, 5, 1, 3, 7, 6, 9})
                        .contiguous()
                        .reshape({M, N});
    t_sf = t_sf.view({M / SMEM_M, N / SMEM_N, SMEM_M / 32, SMEM_N / 64, 8, 4, 4})
                   .permute({0, 2, 5, 4, 1, 3, 6})
                   .contiguous()
                   .reshape({M, N / 16});

    at::Tensor t_f16 =
            (t_fp4_sim.view({M, N / 16, 16}).to(c10::kFloat) * t_sf.unsqueeze(-1).to(c10::kFloat))
                    .reshape({M, N});
    t_f16 = t_f16.to(c10::kHalf);

    return t_f16;
}
#endif  // #if DORADO_CUDA_BUILD

}  // namespace dorado::nn
