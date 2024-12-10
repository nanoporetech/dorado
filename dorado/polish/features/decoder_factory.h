#pragma once

#include "decoder_base.h"
#include "polish/architectures/model_config.h"

#include <memory>
#include <string>

namespace dorado::polisher {

std::unique_ptr<DecoderBase> decoder_factory(const ModelConfig& config);

}  // namespace dorado::polisher