#pragma once

namespace dorado::config {
struct BasecallModelConfig;
}

namespace dorado::basecall {

enum class MetalModelImplementation { mps, lstm };

MetalModelImplementation select_metal_model_implementation(
        const config::BasecallModelConfig& model_config);

}  // namespace dorado::basecall
