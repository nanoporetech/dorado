
#include "ModelFinder.h"

#include "DataLoader.h"
#include "models/kits.h"
#include "models/metadata.h"
#include "models/models.h"
#include "utils/fs_utils.h"
#include "utils/math_utils.h"
#include "utils/string_utils.h"

#include <spdlog/spdlog.h>

#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>

namespace dorado {

using namespace models;
namespace fs = std::filesystem;

ModelSelection ModelComplexParser::parse(const std::string& arg) {
    ModelSelection selection{arg};
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
                spdlog::trace("Model option: '{}' unknown - assuming path", variant_str);
                selection.model = ModelVariantPair{model_variant};
            } else {
                spdlog::trace("'{}' found variant: '{}' and version: '{}'", variant_str,
                              to_string(model_variant), to_string(version));
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
                   [](unsigned char c) { return std::tolower(c); });

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
            throw std::runtime_error("Failed to parse model version: '" + version +
                                     ", invalid part: '" + value + "' - " + e.what());
        }
    }

    while (nums.size() < 3) {
        nums.push_back("0");
    }
    // join parts -> "v0.0.0"
    return "v" + utils::join(nums, ".");
}

ModelFinder::ModelFinder(const Chemistry& chemistry,
                         const ModelSelection& selection,
                         bool suggestions)
        : m_chemistry(chemistry),
          m_selection(selection),
          m_suggestions(suggestions),
          m_simplex_model_info(get_simplex_model_info()){};

fs::path ModelFinder::fetch_simplex_model() {
    return fetch_model(get_simplex_model_name(), "simplex");
}

fs::path ModelFinder::fetch_stereo_model() {
    return fetch_model(get_stereo_model_name(), "stereo duplex");
}

std::vector<fs::path> ModelFinder::fetch_mods_models() {
    const auto model_names = get_mods_model_names();
    std::vector<fs::path> paths;
    for (auto name : model_names) {
        paths.push_back(fetch_model(name, "modification"));
    }
    return paths;
}

fs::path ModelFinder::fetch_model(const std::string& model_name, const std::string& description) {
    const auto local_path = fs::current_path() / model_name;
    if (fs::exists(local_path)) {
        spdlog::debug("Found existing {} model: {}", description, model_name);
        return local_path;
    }

    const fs::path temp_dir = utils::get_downloads_path(std::nullopt);
    const fs::path temp_model_dir = temp_dir / model_name;
    if (models::download_models(temp_dir.u8string(), model_name)) {
        m_downloaded_models.emplace(temp_dir);
    } else {
        throw std::runtime_error("Failed to download + " + description + " model: " + model_name);
    }

    return temp_model_dir;
}

ModelInfo ModelFinder::get_simplex_model_info() const {
    const ModelList& models = simplex_models();
    return find_model(models, "simplex", m_chemistry, m_selection.model, ModsVariantPair(),
                      m_suggestions);
}

std::string ModelFinder::get_simplex_model_name() const { return m_simplex_model_info.name; }

std::vector<std::string> ModelFinder::get_mods_model_names() const {
    const ModelList& models = modified_models();
    std::vector<std::string> model_names;
    for (const auto& mod : m_selection.mods) {
        const auto model_info = find_model(models, "modification", m_chemistry,
                                           m_simplex_model_info.simplex, mod, m_suggestions);
        model_names.push_back(model_info.name);
    }
    return model_names;
}

std::string ModelFinder::get_stereo_model_name() const {
    const ModelList& models = stereo_models();
    const auto model_info = find_model(models, "stereo duplex", m_chemistry,
                                       m_simplex_model_info.simplex, ModsVariantPair(), false);
    return model_info.name;
}

std::vector<std::string> ModelFinder::get_mods_for_simplex_model() const {
    const auto& modification_models = modified_models();
    std::vector<std::string> model_names;
    const auto model_infos =
            find_models(modification_models, m_chemistry, m_selection.model, ModsVariantPair());
    for (const auto& info : model_infos) {
        model_names.push_back(info.name);
    }
    return model_names;
}

Chemistry ModelFinder::inspect_chemistry(const std::string& data, bool recursive_file_loading) {
    const ChemistryMap& kit_map = models::chemistry_map();
    std::vector<ChemistryKey> data_chemistries =
            DataLoader::get_sequencing_chemistry(data, recursive_file_loading);

    std::set<Chemistry> found;
    for (const auto& dc : data_chemistries) {
        const auto it = kit_map.find(dc);
        if (it == kit_map.end()) {
            const auto& [fc, kit, sampling_rate] = dc;
            spdlog::error("No supported chemistry for Flowcell:'{}', Kit:'{}' and Sampling Rate:{}",
                          to_string(fc), to_string(kit), sampling_rate);
            throw std::runtime_error("Could not resolve chemistry from data");
        }
        found.insert(it->second);
    }
    if (found.empty()) {
        throw std::runtime_error("Could not resolve chemistry from data");
    }
    if (found.size() > 1) {
        spdlog::error("Multiple sequencing chemistries found in data");
        for (auto f : found) {
            spdlog::error("Found: {}", to_string(f));
        }
        throw std::runtime_error("Could not uniquely resolve chemistry from inhomogeneous data");
    }
    return *std::begin(found);
}

models::ModelInfo ModelFinder::get_simplex_model_info(const std::string& model_name) {
    return models::get_simplex_model_info(model_name);
}

void check_sampling_rates_compatible(const std::string& model_name,
                                     const std::filesystem::path& data_path,
                                     const int config_sample_rate,
                                     const bool recursive_file_loading) {
    SamplingRate sample_rate;
    SamplingRate data_sample_rate;

    try {
        // Check sample rate of model vs data.
        sample_rate = config_sample_rate < 0 ? get_sample_rate_by_model_name(model_name)
                                             : static_cast<SamplingRate>(config_sample_rate);
        data_sample_rate = DataLoader::get_sample_rate(data_path, recursive_file_loading);
    } catch (const std::exception& e) {
        // Failed the check - warn the user
        spdlog::warn(
                "Could not check that model sampling rate and data sampling rate match. "
                "Proceed with caution. Reason: {}",
                e.what());
        return;
    }

    if (!utils::eq_with_tolerance(data_sample_rate, sample_rate, static_cast<uint16_t>(100))) {
        std::string err = "Sample rate for model (" + std::to_string(sample_rate) + ") and data (" +
                          std::to_string(data_sample_rate) + ") are not compatible.";
        throw std::runtime_error(err);
    }
}

std::vector<std::filesystem::path> get_non_complex_mods_models(
        const std::filesystem::path& simplex_model_path,
        const std::vector<std::string>& mod_bases,
        const std::string& mod_bases_models) {
    if (!mod_bases.empty() && !mod_bases_models.empty()) {
        throw std::runtime_error(
                "CLI arguments --modified-bases and --modified-bases-models are mutually "
                "exclusive");
    }

    std::vector<std::filesystem::path> mods_model_paths;

    if (!mod_bases.empty()) {
        // Foreach --modified-bases get the modified model of that type matched to the simplex model
        std::transform(mod_bases.begin(), mod_bases.end(), std::back_inserter(mods_model_paths),
                       [&simplex_model_path](const std::string& m) {
                           return std::filesystem::path(
                                   models::get_modification_model(simplex_model_path, m));
                       });
    } else if (!mod_bases_models.empty()) {
        // Foreach --modified-bases-models get a path
        const auto split = utils::split(mod_bases_models, ',');
        std::transform(split.begin(), split.end(), std::back_inserter(mods_model_paths),
                       [&](const std::string& m) { return std::filesystem::path(m); });
    }

    return mods_model_paths;
}

}  // namespace dorado