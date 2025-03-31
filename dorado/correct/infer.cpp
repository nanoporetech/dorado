#include "infer.h"

#include "types.h"
#include "utils/memory_utils.h"
#if DORADO_METAL_BUILD
#include "torch_utils/metal_utils.h"
#endif
#if DORADO_CUDA_BUILD
#include "torch_utils/cuda_utils.h"
#endif
#include "utils/string_utils.h"

#include <spdlog/spdlog.h>
#include <toml.hpp>
#include <torch/types.h>

namespace dorado::correction {

int calculate_batch_size(const std::string& device, float memory_fraction) {
    // These sizes are currently hard coded for version 1 model.
    const float model_mem = 1.f;       // GB
    const float per_sample_mem = 1.f;  // GB
    float usable_memory = 0.f;
    if (device == "cpu") {
#if DORADO_METAL_BUILD
        size_t physical_memory =
                utils::get_apple_physical_memory_bytes() / dorado::utils::BYTES_PER_GB;
        usable_memory = physical_memory * memory_fraction;
#else
        size_t free_ram_GB = utils::available_host_memory_GB();
        usable_memory = free_ram_GB * memory_fraction;
#endif
    }
#if DORADO_CUDA_BUILD
    else if (utils::starts_with(device, "cuda")) {
        torch::Device dev = torch::Device(device);
        int64_t available = utils::available_memory(dev) / dorado::utils::BYTES_PER_GB;
        usable_memory = available * memory_fraction;
    }
#endif
    else {
        throw std::runtime_error("Unsupported device: " + device);
    }

    spdlog::debug("Usable memory for dev {}: {} GB", device, usable_memory);
    usable_memory -= model_mem;
    if (usable_memory <= 0.f) {
        return 0;
    } else {
        int batch_size = static_cast<int>(std::round(usable_memory / per_sample_mem));
        // Round down to multiple of 4.
        batch_size = static_cast<int>(batch_size / 4) * 4;
        return batch_size;
    }
}

ModelConfig parse_model_config(const std::filesystem::path& config_path) {
    const toml::value config_toml = toml::parse(config_path.string());

    if (!config_toml.contains("model")) {
        throw std::runtime_error("Model config must include [model] section");
    }

    ModelConfig cfg;

    const auto& model = toml::find(config_toml, "model");
    cfg.version = toml::find<int>(model, "version");
    cfg.window_size = toml::find<int>(model, "window_size");
    cfg.model_type = toml::find<std::string>(model, "model_type");
    cfg.weights_file = toml::find<std::string>(model, "weights_file");
    cfg.model_dir = config_path.parent_path();

    return cfg;
}

}  // namespace dorado::correction
