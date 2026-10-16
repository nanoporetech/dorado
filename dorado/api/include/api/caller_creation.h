#pragma once

#include "basecall/ModelCaller.h"
#include "config/ModBaseModelConfigFwd.h"

#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace dorado::basecall {
class ModelCaller;
}  // namespace dorado::basecall

namespace dorado::modbase {
class ModBaseCaller;
}

namespace dorado::config {
struct BasecallModelConfig;
}

namespace dorado::api {
using dorado::basecall::PipelineType;

std::shared_ptr<basecall::ModelCaller> create_basecall_caller(
        const basecall::BasecallerCreationParams& params);

std::shared_ptr<modbase::ModBaseCaller> create_modbase_caller(
        std::span<const config::ModBaseModelConfig> model_configs,
        int batch_size,
        const std::string& device);

}  // namespace dorado::api
