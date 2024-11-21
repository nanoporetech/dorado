#include "model_config.h"

#include <toml.hpp>
#include <toml/value.hpp>

#include <iostream>

namespace dorado::polisher {

inline void print_toml(const toml::value& val, int indent) {
    const std::string indent_str(indent, ' ');

    if (val.is_table()) {
        const auto& table = val.as_table();
        for (const auto& [key, value] : table) {
            std::cout << indent_str << key << " = ";
            if (value.is_table() || value.is_array()) {
                std::cout << "\n";
            }
            print_toml(value, indent + 2);
        }
    } else if (val.is_array()) {
        const auto& arr = val.as_array();
        std::cout << "[";
        for (size_t i = 0; i < arr.size(); ++i) {
            print_toml(arr[i], 0);
            if (i < arr.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]\n";
    } else {
        // For POD types.
        if (val.is_string()) {
            std::cout << "\"" << val.as_string() << "\"\n";
        } else if (val.is_integer()) {
            std::cout << val.as_integer() << "\n";
        } else if (val.is_floating()) {
            std::cout << val.as_floating() << "\n";
        } else if (val.is_boolean()) {
            std::cout << (val.as_boolean() ? "true" : "false") << "\n";
        } else {
            std::cout << "(unknown type)\n";
        }
    }
}

std::unordered_map<std::string, std::string> parse_kwargs(const toml::value& table) {
    std::unordered_map<std::string, std::string> kwargs;

    if (!table.is_table()) {
        throw std::invalid_argument("Expected a TOML table.");
    }

    for (const auto& [key, value] : table.as_table()) {
        kwargs[key] = toml::format(value);
    }

    return kwargs;
}

ModelConfig parse_model_config(const std::filesystem::path& config_path,
                               const std::string& model_file) {
    const toml::value config_toml = toml::parse(config_path.string());

    if (!config_toml.contains("model")) {
        throw std::runtime_error("Model config must include the [model] section.");
    }
    if (!config_toml.contains("feature_encoder")) {
        throw std::runtime_error("Model config must include the [feature_encoder] section.");
    }
    if (!config_toml.contains("config_version")) {
        throw std::runtime_error("Model config must contain 'config_version' attribute.");
    }

    // print_toml(config_toml, 0);

    ModelConfig cfg;

    // Parse the config version.
    { cfg.version = toml::find<int>(config_toml, "config_version"); }

    // Parse the model info.
    {
        const auto& section = toml::find(config_toml, "model");
        cfg.model_type = toml::find<std::string>(section, "type");
        // cfg.model_file = toml::find<std::string>(section, "model_file");
        cfg.model_file = model_file;
        cfg.model_dir = config_path.parent_path().string();

        // Parse kwargs for the model.
        const auto& model_kwargs = toml::find(section, "kwargs");
        cfg.model_kwargs = parse_kwargs(model_kwargs);
    }

    // Parse the feature encoder info.
    {
        const auto& section = toml::find(config_toml, "feature_encoder");
        cfg.feature_encoder_type = toml::find<std::string>(section, "type");

        // Parse kwargs for the feature extractor.
        const auto& model_kwargs = toml::find(section, "kwargs");
        cfg.feature_encoder_kwargs = parse_kwargs(model_kwargs);

        // Parse dtypes separately and optionally. Perhaps some encoders won't have it.
        // It's easier to parse here than from a string later.
        if (toml::find_or<bool>(section, "kwargs", "dtypes", false)) {
            const auto dtypes_toml =
                    toml::find<std::vector<std::string>>(section, "kwargs", "dtypes");
            cfg.feature_encoder_dtypes =
                    std::vector<std::string>(std::begin(dtypes_toml), std::end(dtypes_toml));
        }
    }

    // Parse the label scheme section.
    {
        const auto& section = toml::find(config_toml, "label_scheme");
        cfg.label_scheme_type = toml::find<std::string>(section, "type");
    }

    return cfg;
}

}  // namespace dorado::polisher
