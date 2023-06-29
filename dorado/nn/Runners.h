#pragma once

#include "ModBaseRunner.h"
#include "ModelRunner.h"
#include "nn/CRFModel.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace dorado {

std::pair<std::vector<dorado::Runner>, size_t> create_basecall_runners(
        const dorado::CRFModelConfig& model_config,
        const std::string& device,
        size_t num_runners,
        size_t batch_size,
        size_t chunk_size,
        float memory_fraction = 1.f,
        bool guard_gpus = false);

std::vector<std::unique_ptr<dorado::ModBaseRunner>> create_modbase_runners(
        const std::string& remora_models,
        const std::string& device,
        size_t remora_runners_per_caller,
        size_t remora_batch_size);

}  // namespace dorado