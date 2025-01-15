#pragma once

#include "cli_utils.h"
#include "model_downloader/model_downloader.h"
#include "models/model_complex.h"
#include "utils/math_utils.h"
#include "utils/string_utils.h"

#include <argparse/argparse.hpp>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace dorado {

namespace model_resolution {

inline models::ModelComplex parse_model_argument(const std::string& model_arg) {
    try {
        return models::ModelComplexParser::parse(model_arg);
    } catch (std::exception& e) {
        spdlog::error("Failed to parse model argument. {}", e.what());
        std::exit(EXIT_FAILURE);
    }
}

inline bool check_model_path(const std::filesystem::path& model_path) noexcept {
    namespace fs = std::filesystem;
    try {
        const auto& p = fs::weakly_canonical(model_path);
        if (!fs::exists(p)) {
            spdlog::error(
                    "Model does not exist at: '{}' - Please download the model or use a model "
                    "complex to automatically download a model",
                    p.string());
            return false;
        }
        if (!fs::is_directory(p)) {
            spdlog::error(
                    "Model is not a directory at: '{}' - Please check your model path or use a "
                    "model complex to automatically download a model.",
                    p.string());
            return false;
        }
        if (fs::is_empty(p)) {
            spdlog::error(
                    "Model is an empty directory at: '{}' - Please check your model path or use a "
                    "model complex to automatically download a model.",
                    p.string());
            return false;
        }
        const auto cfg = p / "config.toml";
        if (!fs::exists(cfg)) {
            spdlog::error(
                    "Model directory is missing a configuration file at: '{}' - Please check your "
                    "model path or download your model again.",
                    cfg.string());
            return false;
        }
    } catch (std::exception& e) {
        spdlog::error("Exception while checking model path at: '{}' - {}", e.what());
        return false;
    }
    return true;
}

// Get the model search directory with the command line argument taking priority over the environment variable.
// Returns std:nullopt if nether are set explicitly
inline std::optional<std::filesystem::path> get_models_directory(
        const argparse::ArgumentParser& parser) {
    const auto models_directory_arg =
            cli::get_optional_argument<std::string>("--models-directory", parser);
    const char* env_path = std::getenv("DORADO_MODELS_DIRECTORY");

    if (models_directory_arg.has_value()) {
        auto path = std::filesystem::path(models_directory_arg.value());
        if (!std::filesystem::exists(path)) {
            spdlog::error("--models-directory path does not exist at: '{}'", path.u8string());
            std::exit(EXIT_FAILURE);
        }
        path = std::filesystem::canonical(path);
        spdlog::debug("Set models directory to: '{}'.", path.u8string());
        return path;
    } else if (env_path != nullptr) {
        auto path = std::filesystem::path(env_path);
        if (!std::filesystem::exists(path)) {
            spdlog::warn(
                    "ignoring environment variable 'DORADO_MODELS_DIRECTORY' - path does not exist "
                    "at: '{}'",
                    path.u8string());
        } else {
            path = std::filesystem::canonical(path);
            spdlog::debug(
                    "set models directory to: '{}' from 'DORADO_MODELS_DIRECTORY' "
                    "environment variable",
                    path.u8string());
            return path;
        }
    }
    return std::nullopt;
}

inline bool mods_model_arguments_valid(const models::ModelComplex& model_complex,
                                       const std::vector<std::string>& mod_bases,
                                       const std::string& mod_bases_models) {
    // Assert that only one of --modified-bases, --modified-bases-models or mods model complex is set
    auto ways = {model_complex.has_mods_variant(), !mod_bases.empty(), !mod_bases_models.empty()};
    if (std::count(ways.begin(), ways.end(), true) > 1) {
        spdlog::error(
                "Only one of --modified-bases, --modified-bases-models, or modified models set "
                "via models argument can be used at once");
        return false;
    };
    return true;
}

inline std::vector<std::filesystem::path> get_non_complex_mods_models(
        const std::filesystem::path& simplex_model_path,
        const std::vector<std::string>& mod_bases,
        const std::string& mod_bases_models,
        model_downloader::ModelDownloader& downloader) {
    if (!mod_bases.empty() && !mod_bases_models.empty()) {
        throw std::runtime_error(
                "CLI arguments --modified-bases and --modified-bases-models are mutually "
                "exclusive");
    }

    std::vector<std::filesystem::path> mods_model_paths;
    if (!mod_bases.empty()) {
        // Foreach --modified-bases get the modified model of that type matched to the simplex model
        std::transform(mod_bases.begin(), mod_bases.end(), std::back_inserter(mods_model_paths),
                       [&simplex_model_path, &downloader](const std::string& m) {
                           const auto mods_info =
                                   models::get_modification_model(simplex_model_path, m);
                           return downloader.get(mods_info, "mods");
                       });
    } else if (!mod_bases_models.empty()) {
        // Foreach --modified-bases-models get a path
        const auto splits = utils::split(mod_bases_models, ',');
        mods_model_paths.reserve(splits.size());
        for (const auto& part : splits) {
            const auto path = std::filesystem::path(part);
            if (!std::filesystem::exists(path)) {
                spdlog::error(
                        "A model path set via --modified-bases-models '{}', does not exist or "
                        "it could not be resolved. All paths must be relative or absolute. "
                        "Please check the modified bases model paths.",
                        part);
            }

            mods_model_paths.emplace_back(path);
        }
    }

    return mods_model_paths;
}

inline void check_sampling_rates_compatible(int model_sample_rate, int data_sample_rate) {
    if (!utils::eq_with_tolerance(data_sample_rate, model_sample_rate, 100)) {
        std::string err = "Sample rate for model (" + std::to_string(model_sample_rate) +
                          ") and data (" + std::to_string(data_sample_rate) +
                          ") are not compatible.";
        throw std::runtime_error(err);
    }
}

}  // namespace model_resolution
}  // namespace dorado