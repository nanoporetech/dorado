#pragma once

#include "basecall/ModelRunnerBase.h"
#include "modbase/ModBaseRunner.h"

#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace dorado {

namespace basecall {
struct CRFModelConfig;

#if DORADO_CUDA_BUILD
class CudaCaller;
#elif DORADO_METAL_BUILD
class MetalCaller;
#endif

}  // namespace basecall

namespace modbase {
class ModBaseCaller;
}

namespace api {

std::pair<std::vector<basecall::RunnerPtr>, size_t> create_basecall_runners(
        const basecall::CRFModelConfig& model_config,
        const std::string& device,
        size_t num_gpu_runners,
        size_t num_cpu_runners,
        size_t batch_size,
        size_t chunk_size,
        float memory_fraction,
        bool guard_gpus);

std::vector<modbase::RunnerPtr> create_modbase_runners(
        const std::vector<std::filesystem::path>& remora_models,
        const std::string& device,
        size_t remora_runners_per_caller,
        size_t remora_batch_size);

#if DORADO_CUDA_BUILD
basecall::RunnerPtr create_basecall_runner(std::shared_ptr<basecall::CudaCaller> caller);
#elif DORADO_METAL_BUILD
basecall::RunnerPtr create_basecall_runner(std::shared_ptr<basecall::MetalCaller> caller);
#endif

modbase::RunnerPtr create_modbase_runner(std::shared_ptr<modbase::ModBaseCaller> caller);

}  // namespace api
}  // namespace dorado
