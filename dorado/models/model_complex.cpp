#include "models/model_complex.h"

#include "models.h"
#include "utils/string_utils.h"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <stdexcept>

namespace dorado::models {

ModelComplex ModelComplexParser::parse(const std::string& arg) {
    ModelComplex selection{arg};
    std::vector<std::pair<std::string, ModelVersion>> items;
    for (const auto& p : utils::split(arg, ',')) {
        items.push_back(parse_model_arg_part(p));
    }

    if (items.empty()) {
        throw std::runtime_error("No model arguments to parse: '" + arg + "'");
    }

    for (size_t idx = 0; idx < items.size(); ++idx) {
        const auto& [variant_str, version] = items.at(idx);
        // Require simplex basecall model is specified first
        if (idx == 0) {
            const auto model_variant = get_model_variant(variant_str);
            if (model_variant == ModelVariant::NONE) {
                spdlog::trace("Model variant: '{}' unknown - assuming path", variant_str);
                selection.model = ModelVariantPair{model_variant};
            } else {
                spdlog::trace("Model complex: '{}' found variant: '{}' and version: '{}'",
                              variant_str, to_string(model_variant), to_string(version));
                selection.model = ModelVariantPair{model_variant, version};
            }
        } else {
            // Remaining models are modification models
            const auto mod_variant = get_mods_variant(variant_str);
            if (mod_variant == ModsVariant::NONE) {
                spdlog::error("Unknown modification variant: '{}' - Choices: {}", variant_str,
                              utils::join(modified_model_variants(), ", "));
                throw std::runtime_error("Failed to parse modified model arguments");
            }
            selection.mods.push_back({mod_variant, version});
        }
    }

    // If the path doesn't exist then issue a warning. The error is caught and handled downstream.
    if (selection.is_path()) {
        // Remove any directory separators from the end
        while (!selection.raw.empty() &&
               (selection.raw.back() == '/' || selection.raw.back() == '\\')) {
            selection.raw.pop_back();
        }

        if (!std::filesystem::exists(std::filesystem::path(selection.raw))) {
            spdlog::warn(
                    "Model argument '{}' did not satisfy the model complex syntax and is assumed "
                    "to be "
                    "a path.",
                    selection.raw);
        }
    }

    return selection;
}

std::pair<std::string, ModelVersion> ModelComplexParser::parse_model_arg_part(
        const std::string& part) {
    std::vector<std::string> sub_parts = utils::split(part, '@');

    if (sub_parts.empty()) {
        throw std::runtime_error("Failed to parse model selection part: '" + part + "'");
    }
    if (sub_parts.size() != 2) {
        return std::pair(part, ModelVersion::NONE);
    }

    // If it's neither a ModelVariant or ModsVariant, it's a path
    const auto& variant = sub_parts.at(0);
    if (get_mods_variant(variant) == ModsVariant::NONE &&
        get_model_variant(variant) == ModelVariant::NONE) {
        // Return the whole part which is now the filepath
        return std::pair(part, ModelVersion::NONE);
    }

    if (sub_parts.at(1) == "latest") {
        return std::pair(variant, ModelVersion::NONE);
    }
    const auto ver = parse_version(sub_parts.at(1));
    const auto& versions = version_map();
    auto it = versions.find(ver);
    if (it == versions.end()) {
        throw std::runtime_error("Version: '" + ver + "' is not an available option");
    }

    return std::pair(variant, it->second);
}

// Given a potentially truncated version string, returns a well-formatted version
// string like "v0.0.0" ensuring that there are 3 values, all values are integers,
// and not empty
std::string ModelComplexParser::parse_version(const std::string& version) {
    auto ver = version;
    // to lower
    std::transform(ver.begin(), ver.end(), ver.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    // assert starts with v/V
    if (!utils::starts_with(ver, "v")) {
        throw std::runtime_error("Failed to parse model version: '" + version + "', '" + ver +
                                 "' - must start with 'v'");
    }

    // Split and check all values and to list of num parts
    std::vector<std::string> nums;
    const auto split_values = utils::split(ver.substr(1), '.');
    for (const std::string& value : split_values) {
        // This will catch trailing .'s and empty parts
        if (value.empty()) {
            spdlog::debug("model version: {} - empty part interpreted as '0'", version);
            nums.push_back("0");
            continue;
        }

        // Asserts that values within periods are integers.
        try {
            if (std::any_of(std::begin(value), std::end(value),
                            [](unsigned char c) { return !std::isdigit(c); })) {
                throw std::runtime_error("Part has non-digit characters");
            }
            nums.push_back(value);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Failed to parse model version: '")
                                             .append(version)
                                             .append(", invalid part: '")
                                             .append(value)
                                             .append("' - ")
                                             .append(e.what()));
        }
    }

    while (nums.size() < 3) {
        nums.push_back("0");
    }
    // join parts -> "v0.0.0"
    return "v" + utils::join(nums, ".");
}

ModelComplexSearch::ModelComplexSearch(const ModelComplex& complex,
                                       Chemistry chemistry,
                                       bool suggestions)
        : m_complex(complex),
          m_chemistry(chemistry),
          m_suggestions(suggestions),
          m_simplex_model_info(resolve_simplex()) {}

ModelInfo ModelComplexSearch::resolve_simplex() const {
    if (m_complex.is_path()) {
        throw std::logic_error(
                "Cannot use model ModelComplexSearch with a simplex model complex which is a path");
    }
    return find_model(simplex_models(), "simplex", m_chemistry, m_complex.model, ModsVariantPair(),
                      m_suggestions);
}

ModelInfo ModelComplexSearch::simplex() const {
    if (m_complex.is_path()) {
        throw std::logic_error(
                "Cannot use model ModelComplexSearch with a simplex model complex which is a path");
    }
    return m_simplex_model_info;
}

ModelInfo ModelComplexSearch::stereo() const {
    return find_model(stereo_models(), "stereo duplex", m_chemistry, m_simplex_model_info.simplex,
                      ModsVariantPair(), false);
}

std::vector<ModelInfo> ModelComplexSearch::simplex_mods() const {
    return find_models(modified_models(), m_chemistry, m_simplex_model_info.simplex,
                       ModsVariantPair());
}

std::vector<ModelInfo> ModelComplexSearch::mods() const {
    std::vector<ModelInfo> models;
    models.reserve(m_complex.mods.size());
    for (const auto& mod : m_complex.mods) {
        const auto model_info = find_model(modified_models(), "modification", m_chemistry,
                                           m_simplex_model_info.simplex, mod, m_suggestions);
        models.push_back(model_info);
    }
    return models;
}

}  // namespace dorado::models
