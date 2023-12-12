#pragma once

#include "compat_utils.h"

#include <torch/torch.h>

namespace dorado::utils {

inline void make_torch_deterministic() {
#if DORADO_GPU_BUILD && !defined __APPLE__
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

}  // namespace dorado::utils
