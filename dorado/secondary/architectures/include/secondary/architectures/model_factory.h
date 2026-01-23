#pragma once

#include "model_config.h"
#include "model_torch_base.h"

#include <memory>
#include <string>

namespace dorado::secondary {

enum class ModelType {
    GRU,
    LATENT_SPACE_LSTM,
    SLOT_ATTENTION_CONSENSUS,
    VARIANT_PERCEIVER,
};

enum class ParameterLoadingStrategy {
    LOAD_WEIGHTS,
    NO_OP,
};

ModelType parse_model_type(const std::string& type);

std::shared_ptr<ModelTorchBase> model_factory(const ModelConfig& config,
                                              const ParameterLoadingStrategy param_strategy);

std::shared_ptr<ModelTorchBase> model_factory(const ModelConfig& config);

}  // namespace dorado::secondary
