#include "nn/KoiUtils.h"

#include <ATen/cuda/CUDAContextLight.h>

namespace {

bool koi_can_use_cutlass_internal(const cudaDeviceProp *const prop) { return (prop->major >= 8); }

bool koi_can_run_flstm_internal(const cudaDeviceProp *const prop) {
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
    cudaDeviceProp *prop = at::cuda::getCurrentDeviceProperties();
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
    cudaDeviceProp *prop = at::cuda::getDeviceProperties(device_id);
    return (prop->major >= 8);
}

}  // namespace dorado::nn
