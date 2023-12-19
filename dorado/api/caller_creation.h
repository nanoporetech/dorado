#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace dorado::basecall {
struct CRFModelConfig;

#if DORADO_GPU_BUILD
#ifndef __APPLE__
class CudaCaller;
#else
class MetalCaller;
#endif
#endif

}  // namespace dorado::basecall

namespace dorado::modbase {
class ModBaseCaller;
}

namespace dorado::callers {

#if DORADO_GPU_BUILD
#ifndef __APPLE__
std::shared_ptr<basecall::CudaCaller> create_cuda_caller(
        const basecall::CRFModelConfig & model_config,
        int chunk_size,
        int batch_size,
        const std::string & device,
        float memory_limit_fraction,
        bool exclusive_gpu_access);
#else
std::shared_ptr<basecall::MetalCaller>
create_metal_caller(const basecall::CRFModelConfig& model_config, int chunk_size, int batch_size);
#endif
#endif

std::shared_ptr<modbase::ModBaseCaller> create_modbase_caller(
        const std::vector<std::filesystem::path> & model_paths,
        int batch_size,
        const std::string & device);

}  // namespace dorado::callers
