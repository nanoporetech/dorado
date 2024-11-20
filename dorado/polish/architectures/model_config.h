#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>

namespace dorado::polisher {

struct ModelConfig {
    int32_t version = 0;

    // Model section.
    std::string model_type;
    std::filesystem::path model_file;
    std::filesystem::path model_dir;
    std::unordered_map<std::string, std::string> model_kwargs;

    // Feature encoder section.
    std::string feature_encoder_type;
    std::unordered_map<std::string, std::string> feature_encoder_kwargs;
    std::vector<std::string> feature_encoder_dtypes;

    // Label scheme section.
    std::string label_scheme_type;
};

ModelConfig parse_model_config(const std::filesystem::path& config_path);

}  // namespace dorado::polisher
