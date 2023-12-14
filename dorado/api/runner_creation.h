#pragma once

#include "basecall/ModelRunner.h"
#include "modbase/ModBaseRunner.h"

#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace dorado {

namespace basecall {
struct CRFModelConfig;
}  // namespace basecall

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

}  // namespace dorado
