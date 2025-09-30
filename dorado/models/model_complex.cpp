#include "models/model_complex.h"

#include "models/metadata.h"
#include "models/models.h"
#include "utils/string_utils.h"

#include <spdlog/spdlog.h>

#include <optional>
#include <stdexcept>
#include <vector>

namespace dorado::models {

ModelComplex::ModelComplex(const std::string& raw,
                           const ModelInfo& simplex,
                           const std::vector<ModelInfo>& mods)
        : m_style(Style::NAMED), m_raw(raw), m_simplex_named(simplex), m_mods_named(mods) {}

ModelComplex::ModelComplex(const std::string& raw,
                           const ModelVariantPair& simplex,
                           const std::vector<ModsVariantPair>& mods)
        : m_style(Style::VARIANT), m_raw(raw), m_simplex_variant(simplex), m_mod_variants(mods) {}

ModelComplex::ModelComplex(const std::string& arg) : m_style(Style::PATH), m_raw(arg) {};

ModelComplex ModelComplex::parse(const std::string& arg) {
    if (arg.empty()) {
        throw std::runtime_error("No model argument");
    }

    const auto named_complex = ModelComplex::parse_names(arg);
    if (named_complex.has_value()) {
        return named_complex.value();
    };

    const auto variant_complex = ModelComplex::parse_variant(arg);
    if (variant_complex.has_value()) {
        return variant_complex.value();
    };

    // Not a NAMED or VARIANT model complex - assume path
    return ModelComplex(arg);
}

std::optional<ModelComplex> ModelComplex::parse_names(const std::string& raw) {
    const auto parts = utils::split(raw, ',');
    ModelInfo simplex;
    std::vector<ModelInfo> mods;
    {
        // The first part can be either SIMPLEX or MODBASE model
        const auto maybe_model_info = try_get_model_info(parts.at(0));
        if (!maybe_model_info.has_value()) {
            return std::nullopt;
        }

        const auto& info = maybe_model_info.value();
        switch (info.model_type) {
        case ModelType::SIMPLEX:
            simplex = info;
            break;
        case ModelType::MODBASE:
            simplex = get_modbase_model_simplex_parent(info);
            mods.push_back(info);
            break;
        default:
            spdlog::error(
                    "Failed to parse basecaller model complex '{}' - a '{}' model is not valid "
                    "here.",
                    raw, to_string(info.model_type));
            throw std::runtime_error("Invalid model argument");
        }
    }

    // All remaining parts must be MODBASE model with the same condition, variant and version as the first.
    for (size_t i = 1; i < parts.size(); ++i) {
        const auto maybe_modbase_info = try_get_model_info(parts.at(i));
        if (!maybe_modbase_info.has_value()) {
            spdlog::error(
                    "Failed to parse model complex '{}'. '{}' is not a recognised model "
                    "name.",
                    raw, parts.at(i));
            throw std::runtime_error("Invalid model argument");
        }

        const auto& info = maybe_modbase_info.value();
        if (info.model_type != ModelType::MODBASE) {
            spdlog::error(
                    "Failed to parse model complex '{}' - Additional models must be modbase models "
                    "but found a '{}' model from '{}'",
                    raw, to_string(info.model_type), parts.at(i));
            throw std::runtime_error("Invalid model argument");
        }

        const bool match_parent = info.chemistry == simplex.chemistry &&
                                  info.simplex.variant == simplex.simplex.variant &&
                                  info.simplex.ver == simplex.simplex.ver;
        if (!match_parent) {
            spdlog::error(
                    "Failed to parse model complex '{}' - Additional modbase model '{}' must work "
                    "with simplex model '{}'.",
                    raw, parts.at(i), simplex.name);
            throw std::runtime_error("Invalid model argument");
        }

        for (const ModelInfo& mod : mods) {
            if (info.name == mod.name) {
                spdlog::error("Failed to parse model complex '{}' - Duplicate modbase model '{}'",
                              raw, parts.at(i));
                throw std::runtime_error("Invalid model argument");
            }
        }
        mods.push_back(info);
    }

    return ModelComplex(raw, simplex, mods);
}

std::optional<ModelComplex> ModelComplex::parse_variant(const std::string& raw) {
    std::vector<std::pair<std::string, ModelVersion>> parts;
    for (const auto& p : utils::split(raw, ',')) {
        parts.push_back(parse_variant_part(p));
    }

    ModelVariantPair simplex;
    std::vector<ModsVariantPair> mods;

    for (size_t idx = 0; idx < parts.size(); ++idx) {
        const auto& [part_str, version] = parts.at(idx);
        // Require simplex basecall model is specified first
        if (idx == 0) {
            const auto model_variant = get_model_variant(part_str);
            if (model_variant == ModelVariant::NONE) {
                return std::nullopt;
            } else {
                spdlog::trace("Model complex '{}' found variant: '{}' and version: '{}'", part_str,
                              to_string(model_variant), to_string(version));
                simplex = ModelVariantPair{model_variant, version};
            }
        } else {
            // Remaining models are modification models
            const auto mod_variant = get_mods_variant(part_str);
            if (mod_variant == ModsVariant::NONE) {
                spdlog::error(
                        "Model complex '{}' has unknown modification variant: '{}' "
                        "- Choices: {}",
                        raw, part_str, utils::join(modified_model_variants(), ", "));
                throw std::runtime_error("Failed to parse modified model arguments");
            }
            mods.push_back({mod_variant, version});
        }
    }

    return ModelComplex(raw, simplex, mods);
};

std::pair<std::string, ModelVersion> ModelComplex::parse_variant_part(const std::string& part) {
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
    const auto ver = parse_variant_version(sub_parts.at(1));
    const auto& versions = version_map();
    auto it = versions.find(ver);
    if (it == versions.end()) {
        throw std::runtime_error("Version: '" + ver + "' is not an available option");
    }

    return std::pair(variant, it->second);
}

std::string ModelComplex::parse_variant_version(const std::string& version) {
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

// Only valid IFF style::NAMED
const ModelInfo& ModelComplex::get_named_simplex_model() const {
    if (!m_simplex_named.has_value() || m_style != Style::NAMED) {
        throw std::logic_error("Invalid call to ModelComplex::get_named_simplex_models");
    }
    return m_simplex_named.value();
}

const std::vector<ModelInfo>& ModelComplex::get_named_mods_models() const { return m_mods_named; }

// Only valid IFF style::VARIANT
const ModelVariantPair& ModelComplex::get_simplex_model_variant() const {
    if (!m_simplex_variant.has_value() || m_style != Style::VARIANT) {
        throw std::logic_error("Invalid call to ModelComplex::get_simplex_model_variant.");
    }
    return m_simplex_variant.value();
}

const std::vector<ModsVariantPair>& ModelComplex::get_mod_model_variants() const {
    return m_mod_variants;
}
}  // namespace dorado::models
