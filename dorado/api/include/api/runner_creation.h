#pragma once

#include "api/caller_creation.h"
#include "basecall/ModelRunnerBase.h"
#include "config/ModBaseModelConfigFwd.h"
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
        const std::vector<config::ModBaseModelConfig>& model_configs,
        const std::string& device,
        size_t runners_per_caller,
        size_t batch_size);

size_t get_num_batch_dims(const std::shared_ptr<basecall::ModelCaller>& caller);
basecall::RunnerPtr create_basecall_runner(std::shared_ptr<basecall::ModelCaller> caller,
                                           size_t batch_dims_idx);

modbase::RunnerPtr create_modbase_runner(std::shared_ptr<modbase::ModBaseCaller> caller);

}  // namespace dorado::api
