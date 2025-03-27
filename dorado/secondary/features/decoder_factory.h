#pragma once

#include "decoder_base.h"

#include <memory>

namespace dorado::secondary {

struct ModelConfig;

std::unique_ptr<DecoderBase> decoder_factory(const ModelConfig& config);

}  // namespace dorado::secondary
