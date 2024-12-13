#pragma once

#include "decoder_base.h"

#include <memory>

namespace dorado::polisher {

struct ModelConfig;

std::unique_ptr<DecoderBase> decoder_factory(const ModelConfig& config);

}  // namespace dorado::polisher