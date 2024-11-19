#pragma once

#if DORADO_CUDA_BUILD
#include "torch_utils/gpu_monitor.h"

#include <torch/cuda.h>
#endif

#include <string>

namespace dorado::utils {

inline std::string get_auto_detected_device() {
#if DORADO_METAL_BUILD
    return "metal";
#elif DORADO_CUDA_BUILD
    // Using get_device_count will force a wait for NVML to load, which will ensure the driver has started up.
    return utils::gpu_monitor::get_device_count() > 0 ? "cuda:all" : "cpu";
#else
    return "cpu";
#endif
}

}  // namespace dorado::utils