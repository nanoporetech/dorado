#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace dorado::basecall {
struct CRFModelConfig;

#if DORADO_CUDA_BUILD
class CudaCaller;
#elif DORADO_METAL_BUILD
class MetalCaller;
#endif

}  // namespace dorado::basecall

namespace dorado::modbase {
class ModBaseCaller;
}

namespace dorado::api {

#if DORADO_CUDA_BUILD
std::shared_ptr<basecall::CudaCaller> create_cuda_caller(
        const basecall::CRFModelConfig & model_config,
        int chunk_size,
        int batch_size,
        const std::string & device,
        float memory_limit_fraction,
        bool exclusive_gpu_access);
#elif DORADO_METAL_BUILD
std::shared_ptr<basecall::MetalCaller> create_metal_caller(
        const basecall::CRFModelConfig& model_config,
        int chunk_size,
        int batch_size,
        float memory_limit_fraction);
#endif

std::shared_ptr<modbase::ModBaseCaller> create_modbase_caller(
        const std::vector<std::filesystem::path> & model_paths,
        int batch_size,
        const std::string & device);

}  // namespace dorado::api
