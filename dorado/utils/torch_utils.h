#pragma once

#include "compat_utils.h"

#if DORADO_CUDA_BUILD
#include <c10/cuda/CUDACachingAllocator.h>
#endif
#include <torch/torch.h>

namespace dorado::utils {

inline void make_torch_deterministic() {
#if DORADO_CUDA_BUILD
    setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", true);
    torch::globalContext().setDeterministicCuDNN(true);
    torch::globalContext().setBenchmarkCuDNN(false);
#endif

#if TORCH_VERSION_MAJOR > 1 || TORCH_VERSION_MINOR >= 11
    torch::globalContext().setDeterministicAlgorithms(true, false);
#else
    torch::globalContext().setDeterministicAlgorithms(true);
#endif
}

inline void set_torch_allocator_max_split_size() {
#if DORADO_CUDA_BUILD && TORCH_VERSION_MAJOR >= 2
    const char *pytorch_cuda_alloc_conf = std::getenv("PYTORCH_CUDA_ALLOC_CONF");
    if (pytorch_cuda_alloc_conf != nullptr) {
        std::string_view str(pytorch_cuda_alloc_conf);
        if (str.find("max_split_size_mb") != std::string::npos) {
            // user has set this via env_var - let torch parse and use their value
            return;
        }
    }

    // Do not re-use smaller chunks of large buffers
    // This prevents small allocations from reusing large sections of cached allocated memory
    // which can lead to OoM errors when the original large allocation is needed again
    auto max_split_size_mb = 25;
    std::string max_split_size_mb_settings =
            "max_split_size_mb:" + std::to_string(max_split_size_mb);
    c10::cuda::CUDACachingAllocator::setAllocatorSettings(max_split_size_mb_settings);
#endif
}

}  // namespace dorado::utils
