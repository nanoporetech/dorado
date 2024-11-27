#pragma once

#include "feature_decoder.h"
#include "model_config.h"

#include <memory>
#include <string>

namespace dorado::polisher {

std::unique_ptr<FeatureDecoder> decoder_factory(const ModelConfig& config);

}  // namespace dorado::polisher