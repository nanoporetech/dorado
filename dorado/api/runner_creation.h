#pragma once

#include "basecall/ModelRunnerBase.h"
#include "caller_creation.h"
#include "modbase/ModBaseRunner.h"

#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace dorado::api {

std::pair<std::vector<basecall::RunnerPtr>, size_t> create_basecall_runners(
        const basecall::CRFModelConfig& model_config,
        const std::string& device,
        size_t num_gpu_runners,
        size_t num_cpu_runners,
        float memory_fraction,
        PipelineType pipeline_type,
        float batch_size_time_penalty);

std::vector<modbase::RunnerPtr> create_modbase_runners(
        const std::vector<std::filesystem::path>& remora_models,
        const std::string& device,
        size_t remora_runners_per_caller,
        size_t remora_batch_size);

#if DORADO_CUDA_BUILD
size_t get_num_batch_dims(const std::shared_ptr<basecall::CudaCaller>& caller);
basecall::RunnerPtr create_basecall_runner(std::shared_ptr<basecall::CudaCaller> caller,
                                           size_t batch_dims_idx);
#elif DORADO_METAL_BUILD
basecall::RunnerPtr create_basecall_runner(std::shared_ptr<basecall::MetalCaller> caller);
#endif

modbase::RunnerPtr create_modbase_runner(std::shared_ptr<modbase::ModBaseCaller> caller);

}  // namespace dorado::api
