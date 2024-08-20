#pragma once

#if DORADO_CUDA_BUILD
#include <torch/cuda.h>
#endif

#include <string>

namespace dorado::utils {

inline std::string get_auto_detected_device() {
#if DORADO_METAL_BUILD
    return "metal";
#elif DORADO_CUDA_BUILD
    return torch::cuda::is_available() ? "cuda:all" : "cpu";
#else
    return "cpu";
#endif
}

}  // namespace dorado::utils