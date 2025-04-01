#include "model_config.h"

#include <toml.hpp>

#include <cstddef>
#include <ostream>
#include <stdexcept>

namespace dorado::secondary {

namespace {

void print_toml(std::ostream& os, const toml::value& val, int indent) {
    const std::string indent_str(indent, ' ');

    if (val.is_table()) {
        const auto& table = val.as_table();
        for (const auto& [key, value] : table) {
            os << indent_str << key << " = ";
            if (value.is_table() || value.is_array()) {
                os << '\n';
            }
            print_toml(os, value, indent + 2);
        }
    } else if (val.is_array()) {
        const auto& arr = val.as_array();
        os << "[";
        for (size_t i = 0; i < std::size(arr); ++i) {
            print_toml(os, arr[i], 0);
            if ((i + 1) < std::size(arr)) {
                os << ", ";
            }
        }
        os << "]\n";
    } else {
        // For POD types.
        if (val.is_string()) {
            os << "\"" << val.as_string() << "\"\n";
        } else if (val.is_integer()) {
            os << val.as_integer() << '\n';
        } else if (val.is_floating()) {
            os << val.as_floating() << '\n';
        } else if (val.is_boolean()) {
            os << (val.as_boolean() ? "true" : "false") << '\n';
        } else {
            os << "(unknown type)\n";
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

}  // namespace

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
    (void)&print_toml;

    ModelConfig cfg;

    // Parse the config version.
    {
        cfg.version = toml::find<int>(config_toml, "config_version");
    }

    // Parse the config version.
    {
        cfg.basecaller_model = toml::find<std::string>(config_toml, "basecaller_model");
    }

    // Check if the "supported_basecallers" key exists
    {
        cfg.supported_basecallers.emplace(cfg.basecaller_model);

        if (config_toml.contains("supported_basecallers")) {
            try {
                std::vector<std::string> data =
                        toml::find<std::vector<std::string>>(config_toml, "supported_basecallers");
                cfg.supported_basecallers.insert(std::begin(data), std::end(data));
            } catch (const std::exception& e) {
                throw std::runtime_error("Error parsing 'supported_basecallers' from config: '" +
                                         config_path.string() + "'. Message: " + e.what());
            }
        }
    }

    // Parse the model info.
    {
        const auto& section = toml::find(config_toml, "model");
        cfg.model_type = toml::find<std::string>(section, "type");
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
        if (cfg.feature_encoder_kwargs.find("dtypes") != std::end(cfg.feature_encoder_kwargs)) {
            const auto dtypes_toml = toml::find<std::vector<std::string>>(model_kwargs, "dtypes");
            cfg.feature_encoder_dtypes =
                    std::vector<std::string>(std::begin(dtypes_toml), std::end(dtypes_toml));

            // The convention in Medaka is that when num_dtypes == 1, then the dtypes array is empty.
            // However, it is represented as actually an array of one element consisting of an empty string.
            // In Dorado, we simply use an empty vector for no custom data types.
            if (cfg.feature_encoder_dtypes == std::vector<std::string>{""}) {
                cfg.feature_encoder_dtypes.clear();
            }
        }
    }

    // Parse the label scheme section.
    {
        const auto& section = toml::find(config_toml, "label_scheme");
        cfg.label_scheme_type = toml::find<std::string>(section, "type");
    }

    return cfg;
}

}  // namespace dorado::secondary
