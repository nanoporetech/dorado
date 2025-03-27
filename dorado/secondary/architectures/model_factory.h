#pragma once

#include "model_config.h"
#include "model_gru.h"
#include "model_latent_space_lstm.h"
#include "model_torch_base.h"
#include "model_torch_script.h"

#include <memory>
#include <string>

namespace dorado::secondary {

enum class ModelType {
    GRU,
    LATENT_SPACE_LSTM,
};

ModelType parse_model_type(const std::string& type);

std::shared_ptr<ModelTorchBase> model_factory(const ModelConfig& config);

}  // namespace dorado::secondary
