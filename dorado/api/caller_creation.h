#pragma once

#include <memory>
#include <string>

namespace dorado::basecall {
struct CRFModelConfig;

#if DORADO_GPU_BUILD && !defined(__APPLE__)
class CudaCaller;
#endif

}  // namespace dorado::basecall

namespace dorado::callers {

#if DORADO_GPU_BUILD && !defined(__APPLE__)
std::shared_ptr<basecall::CudaCaller> create_cuda_caller(
        const basecall::CRFModelConfig &model_config,
        int chunk_size,
        int batch_size,
        const std::string &device,
        float memory_limit_fraction,
        bool exclusive_gpu_access);
#endif

}  // namespace dorado::callers
