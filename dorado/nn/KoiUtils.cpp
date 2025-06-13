#include "nn/KoiUtils.h"

#include <ATen/cuda/CUDAContext.h>

namespace dorado::nn {

// TODO: These should really be part of Koi
bool koi_can_use_cutlass() {
    cudaDeviceProp *prop = at::cuda::getCurrentDeviceProperties();
    return (prop->major >= 8);
}
bool koi_can_use_quantised_lstm() {
    cudaDeviceProp *prop = at::cuda::getCurrentDeviceProperties();
    // DP4A is supported on Pascal and later, except for TX2 (sm_62).
    return (prop->major > 6) || (prop->major == 6 && prop->minor != 2);
}

}  // namespace dorado::nn
