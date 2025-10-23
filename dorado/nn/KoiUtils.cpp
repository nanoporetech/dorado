#include "nn/KoiUtils.h"

#include "koi_thread_pool.h"

#include <ATen/cuda/CUDAContext.h>
#include <spdlog/spdlog.h>

#include <stdexcept>

namespace dorado::nn {

// TODO: These should really be part of Koi
bool koi_can_use_cutlass() {
    cudaDeviceProp *prop = at::cuda::getCurrentDeviceProperties();
    return (prop->major >= 8);
}
bool koi_can_use_cutlass(const int device_id) {
    const cudaDeviceProp *const prop = at::cuda::getDeviceProperties(device_id);
    return (prop->major >= 8);
}
bool koi_can_use_quantised_lstm() {
    cudaDeviceProp *prop = at::cuda::getCurrentDeviceProperties();
    // DP4A is supported on Pascal and later, except for TX2 (sm_62).
    return (prop->major > 6) || (prop->major == 6 && prop->minor != 2);
}

KoiThreads::KoiThreads(int num_threads) {
    const int ret = koi_create_thread_pool(&m_threads, num_threads);
    if (ret != KOI_SUCCESS) {
        throw std::runtime_error("Failed to create koi thread pool");
    }
}

KoiThreads::~KoiThreads() {
    const int ret = koi_destroy_thread_pool(&m_threads);
    if (ret != KOI_SUCCESS) {
        spdlog::error("Failed to destroy koi thread pool");
    }
}

}  // namespace dorado::nn
