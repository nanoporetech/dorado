#include "ModelCallerSelection.h"

#include "config/BasecallModelConfig.h"

#include <stdexcept>

namespace dorado::basecall {

MetalModelImplementation select_metal_model_implementation(
        const config::BasecallModelConfig& model_config) {
    if (model_config.is_tx_model() || model_config.is_flstm_model()) {
        return MetalModelImplementation::mps;
    }
    if (model_config.is_lstm_model()) {
        return MetalModelImplementation::lstm;
    }
    throw std::runtime_error("No metal caller available for model: " + model_config.model_name());
}

}  // namespace dorado::basecall
