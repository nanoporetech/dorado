#pragma once

#include "gru_model.h"
#include "model_config.h"
#include "torch_model_base.h"
#include "torch_script_model.h"

#include <string>

#ifdef NDEBUG
#define LOG_TRACE(...)
#else
#define LOG_TRACE(...) spdlog::trace(__VA_ARGS__)
#endif

namespace dorado::polisher {

enum class ModelType {
    GRU,
};

ModelType parse_model_type(const std::string& type);

std::shared_ptr<TorchModel> model_factory(const ModelConfig& config);

}  // namespace dorado::polisher