#pragma once

#include "counts_feature_encoder.h"
#include "gru_model.h"
#include "torch_model_base.h"

#ifdef NDEBUG
#define LOG_TRACE(...)
#else
#define LOG_TRACE(...) spdlog::trace(__VA_ARGS__)
#endif

namespace dorado::polisher {

class PolishArchitecture {};

}  // namespace dorado::polisher