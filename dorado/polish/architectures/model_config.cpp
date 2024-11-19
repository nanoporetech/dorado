#include "model_config.h"

#include "utils/memory_utils.h"
#include "utils/string_utils.h"

#include <spdlog/spdlog.h>
#include <toml.hpp>
#include <toml/value.hpp>

#include <iostream>

namespace dorado::polisher {

// void print_toml(const toml::value& val, int indent = 0) {
//     // Add indentation based on the level
//     std::string indent_str(indent, ' ');

//     if (val.is_table()) {
//         const auto& table = val.as_table();
//         for (const auto& [key, value] : table) {
//             std::cout << indent_str << key << " = ";
//             if (value.is_table() || value.is_array()) {
//                 std::cout << "\n";
//             }
//             print_toml(value, indent + 2);  // Increase indentation for nested tables/arrays
//         }
//     } else if (val.is_array()) {
//         const auto& arr = val.as_array();
//         std::cout << "[";
//         for (size_t i = 0; i < arr.size(); ++i) {
//             print_toml(arr[i], 0);
//             if (i < arr.size() - 1) {
//                 std::cout << ", ";
//             }
//         }
//         std::cout << "]\n";
//     } else {
//         // For primitive types (strings, ints, etc.)
//         if (val.is_string()) {
//             std::cout << "\"" << val.as_string() << "\"\n";
//         } else if (val.is_integer()) {
//             std::cout << val.as_integer() << "\n";
//         } else if (val.is_floating()) {
//             std::cout << val.as_floating() << "\n";
//         } else if (val.is_boolean()) {
//             std::cout << (val.as_boolean() ? "true" : "false") << "\n";
//             // } else if (val.is_none()) {
//             //     std::cout << "null\n";
//         }
//     }
// }

ModelConfig parse_model_config(const std::filesystem::path& config_path) {
    const toml::value config_toml = toml::parse(config_path.string());

    if (!config_toml.contains("model")) {
        throw std::runtime_error("Model config must include [model] section");
    }
    if (!config_toml.contains("feature_encoder")) {
        throw std::runtime_error("Model config must include [feature_encoder] section");
    }

    // print_toml(config_toml);

    ModelConfig cfg;

    const auto& model = toml::find(config_toml, "model");
    cfg.version = toml::find<int>(model, "version");
    cfg.model_type = toml::find<std::string>(model, "type");
    cfg.model_file = toml::find<std::string>(model, "model_file");
    cfg.model_dir = config_path.parent_path().string();

    const auto& feature_encoder = toml::find(config_toml, "feature_encoder");
    cfg.feature_encoder_type = toml::find<std::string>(feature_encoder, "type");

    // cfg.feature_vector_length = toml::find<int32_t>(feature_encoder, "feature_vector_length");
    // cfg.feature_encoder_normalise = toml::find<std::string>(feature_encoder, "normalise");
    // cfg.feature_encoder_tag_keep_missing = toml::find<bool>(feature_encoder, "tag_keep_missing");

    // dtypes
    // feature_indices

    return cfg;
}

}  // namespace dorado::polisher
