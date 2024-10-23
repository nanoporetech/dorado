#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#ifdef NDEBUG
#define LOG_TRACE(...)
#else
#define LOG_TRACE(...) spdlog::trace(__VA_ARGS__)
#endif

namespace dorado::polisher {

struct ModelConfig {
    int32_t version = 0;
    std::string model_type;
    std::string model_file;
    std::string model_dir;

    std::string feature_encoder_type;
    // int32_t feature_vector_length = 0;
    // std::string feature_encoder_normalise;
    // bool feature_encoder_tag_keep_missing = false;
};

ModelConfig parse_model_config(const std::filesystem::path& config_path);

std::string resolve_model_name(const std::string& model_name);

}  // namespace dorado::polisher
