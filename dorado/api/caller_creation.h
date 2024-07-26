#pragma once

#include "basecall/ModelRunnerBase.h"

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
using dorado::basecall::PipelineType;

#if DORADO_CUDA_BUILD
std::shared_ptr<basecall::CudaCaller> create_cuda_caller(
        const basecall::CRFModelConfig& model_config,
        const std::string& device,
        float memory_limit_fraction,
        PipelineType pipeline_type,
        float batch_size_time_penalty,
        bool run_batchsize_benchmarks,
        bool emit_batchsize_benchmarks);
#elif DORADO_METAL_BUILD
std::shared_ptr<basecall::MetalCaller> create_metal_caller(
        const basecall::CRFModelConfig& model_config,
        float memory_limit_fraction);
#endif

std::shared_ptr<modbase::ModBaseCaller> create_modbase_caller(
        const std::vector<std::filesystem::path>& model_paths,
        int batch_size,
        const std::string& device);

}  // namespace dorado::api
