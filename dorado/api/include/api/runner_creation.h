#pragma once

#include "api/caller_creation.h"
#include "basecall/ModelRunnerBase.h"
#include "modbase/ModBaseRunner.h"

#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace dorado::api {

bool check_variable_chunk_sizes_supported(const config::BasecallModelConfig& model_config,
                                          std::span<const int> device_ids);

std::pair<std::vector<basecall::RunnerPtr>, size_t> create_basecall_runners(
        const basecall::BasecallerCreationParams& params,
        size_t num_gpu_runners,
        size_t num_cpu_runners);

std::vector<modbase::RunnerPtr> create_modbase_runners(
        const std::vector<std::filesystem::path>& modbase_models,
        const std::string& device,
        size_t runners_per_caller,
        size_t batch_size);

#if DORADO_CUDA_BUILD
size_t get_num_batch_dims(const std::shared_ptr<basecall::CudaCaller>& caller);
basecall::RunnerPtr create_basecall_runner(std::shared_ptr<basecall::CudaCaller> caller,
                                           size_t batch_dims_idx);
#elif DORADO_METAL_BUILD
size_t get_num_batch_dims(const std::shared_ptr<basecall::MetalCaller>& caller);
basecall::RunnerPtr create_basecall_runner(std::shared_ptr<basecall::MetalCaller> caller,
                                           size_t batch_dims_idx);
#endif

modbase::RunnerPtr create_modbase_runner(std::shared_ptr<modbase::ModBaseCaller> caller);

}  // namespace dorado::api
