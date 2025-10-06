#include "model_resolver/ModelResolver.h"

#include "config/BasecallModelConfig.h"
#include "file_info/file_info.h"
#include "model_downloader/model_downloader.h"
#include "models/kits.h"
#include "models/metadata.h"
#include "models/model_complex.h"
#include "models/models.h"
#include "utils/math_utils.h"
#include "utils/string_utils.h"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <vector>

namespace dorado::model_resolution {
using namespace models;
namespace fs = std::filesystem;

namespace {

std::optional<ModelInfo> get_model_info_from_path(const std::filesystem::path& path) {
    const auto& name = path.filename().string();
    try {
        return models::get_model_info(name);
    } catch (const std::exception& e) {
        spdlog::debug("Failed to get ModelInfo from path '{}' - '{}'", name, e.what());
        return std::nullopt;
    }
}

}  // namespace

// Get the model search directory with the command line argument taking priority over the environment variable.
// Returns std:nullopt if nether are set explicitly
std::optional<fs::path> get_models_directory(
        const std::optional<std::string>& models_directory_arg) {
    const char* env_path = std::getenv("DORADO_MODELS_DIRECTORY");

    if (models_directory_arg.has_value()) {
        auto path = fs::path(models_directory_arg.value());
        if (!fs::exists(path)) {
            throw std::runtime_error(
                    fmt::format("--models-directory path does not exist at: '{}'.", path.string()));
        }
        path = fs::canonical(path);
        spdlog::trace("Set models directory to: '{}'.", path.string());
        return path;
    } else if (env_path != nullptr) {
        auto path = fs::path(env_path);
        if (!fs::exists(path)) {
            spdlog::warn(
                    "Ignoring environment variable 'DORADO_MODELS_DIRECTORY' - path does not exist "
                    "at: '{}'.",
                    path.string());
        } else {
            path = fs::canonical(path);
            spdlog::trace(
                    "Set models directory to: '{}' from 'DORADO_MODELS_DIRECTORY' "
                    "environment variable.",
                    path.string());
            return path;
        }
    }
    return std::nullopt;
}

ModelResolver::ModelResolver(Mode mode,
                             std::string model_complex,
                             std::string modbase_models,
                             std::vector<std::string> modbases,
                             std::optional<std::string> stereo,
                             std::optional<std::string> models_directory,
                             bool skip_model_compatibility_check,
                             const std::vector<std::filesystem::directory_entry>& reads)
        : m_mode(mode),
          m_model_complex(std::move(model_complex)),
          m_modbase_models(std::move(modbase_models)),
          m_modbases(std::move(modbases)),
          m_stereo(std::move(stereo)),
          m_models_directory_arg(std::move(models_directory)),
          m_reads(reads),
          m_skip_model_compatibility_check(skip_model_compatibility_check) {}

ModelSources ModelResolver::resolve() {
    m_models_directory = get_models_directory(m_models_directory_arg);

    if (m_model_complex.empty()) {
        throw std::runtime_error("Model argument must not be empty.");
    }

    ModelSources model_sources = resolve_model_complex();
    resolve_modbase_models(model_sources);
    resolve_stereo_models(model_sources);

    if (m_check_paths_override && !model_sources.check_paths()) {
        throw std::runtime_error("Model validity check failed.");
    }

    if (!m_skip_model_compatibility_check) {
        model_compatibility_check(model_sources);
    }

    if (m_mode != Mode::DOWNLOADER) {
        warn_rna_model(model_sources.simplex);
        warn_stereo_fast(model_sources.stereo);
    }

    return model_sources;
};

ModelSources ModelResolver::resolve_model_complex() const {
    const auto model_complex = ModelComplex::parse(m_model_complex);

    // Downloader only supports named models or variants
    if (model_complex.is_path_style() && m_mode != Mode::DOWNLOADER) {
        const auto path = fs::weakly_canonical(fs::path(model_complex.get_raw()));
        if (!check_model_path(path, true)) {
            throw std::runtime_error("Failed to load model from path.");
        }

        const ModelSource simplex{path, get_model_info_from_path(path), false};
        return ModelSources{simplex, {}, std::nullopt};
    }

    if (model_complex.is_named_style()) {
        const ModelInfo& simplex_info = model_complex.get_named_simplex_model();
        const ModelSource simplex = find_or_download_model(simplex_info);

        std::vector<ModelSource> mods;
        for (const ModelInfo& mod : model_complex.get_named_mods_models()) {
            mods.push_back(find_or_download_model(mod));
        }

        return ModelSources{simplex, std::move(mods), std::nullopt};
    }

    if (model_complex.is_variant_style()) {
        const Chemistry chemistry = get_chemistry();
        const ModelVariantPair simplex_variant = model_complex.get_simplex_model_variant();
        const ModelInfo simplex_info = find_model(simplex_models(), "simplex", chemistry,
                                                  simplex_variant, ModsVariantPair(), true);
        const ModelSource simplex = find_or_download_model(simplex_info);

        std::vector<ModelSource> mods;
        for (const ModsVariantPair& mod_variant : model_complex.get_mod_model_variants()) {
            const ModelInfo modbase_info = find_model(modified_models(), "modbase", chemistry,
                                                      simplex_variant, mod_variant, true);
            mods.push_back(find_or_download_model(modbase_info));
        }

        return ModelSources{simplex, std::move(mods), std::nullopt};
    }

    throw std::logic_error("Failed to resolve Model Complex: '" + m_model_complex + "'.");
}

void ModelResolver::resolve_modbase_models(ModelSources& model_sources) const {
    const bool used_complex = !model_sources.mods.empty();
    const bool used_paths = !m_modbase_models.empty();
    const bool used_modbases = !m_modbases.empty();

    if (used_complex && (used_paths || used_modbases)) {
        throw std::logic_error(
                "Modbase models set via model complex, --modified-bases and "
                "--modified-bases-models are mutually exclusive.");
    }

    const auto& simplex = model_sources.simplex;

    if (used_modbases) {
        model_sources.mods.reserve(m_modbases.size());
        for (const auto& modbase_arg : m_modbases) {
            const auto modbase_varint = get_mods_variant(modbase_arg);
            const auto simplex_name = simplex.info.has_value() ? simplex.info->name
                                                               : simplex.path.filename().string();
            const auto mods_info = get_modification_model(
                    simplex_name, ModsVariantPair{modbase_varint, ModelVersion::NONE});

            if (simplex.info.has_value() && (simplex.info->simplex != mods_info.simplex)) {
                throw std::runtime_error(
                        fmt::format("Modbases model '{}' is incompatible with simplex model '{}'",
                                    mods_info.name, simplex.info->name));
            }

            for (const auto& prev : model_sources.mods) {
                if (prev.info->name == mods_info.name) {
                    throw std::runtime_error(
                            fmt::format("Duplicate modbases model found: '{}'", prev.info->name));
                }
            }

            const auto mods_source = find_or_download_model(mods_info);
            model_sources.mods.push_back(mods_source);
        }
    }

    if (used_paths) {
        const auto splits = utils::split(m_modbase_models, ',');
        model_sources.mods.reserve(splits.size());
        for (const auto& part : splits) {
            const auto path = fs::path(part);
            const auto maybe_info = get_model_info_from_path(path);

            for (const auto& prev : model_sources.mods) {
                if (prev.path.filename().string() == path.filename().string()) {
                    throw std::runtime_error(fmt::format("Duplicate modbases model found: '{}'",
                                                         path.filename().string()));
                }
            }

            model_sources.mods.push_back(ModelSource{path, maybe_info, false});
        }
    }
}

void ModelResolver::resolve_stereo_models(ModelSources& model_sources) const {
    if (m_mode != Mode::DUPLEX) {
        // No stereo model requested
        return;
    }

    if (m_stereo.has_value() && !m_stereo->empty()) {
        // Use the stero model path provided
        const auto path = fs::path(m_stereo.value());
        if (!fs::exists(path)) {
            throw std::runtime_error(
                    fmt::format("--stereo-model does not exist at: '{}'.", path.string()));
        }

        const auto maybe_info = get_model_info_from_path(path);
        model_sources.stereo = ModelSource{path, maybe_info, false};
        return;
    }

    // Resolve the stereo model from the simplex model info
    const auto& simplex = model_sources.simplex;
    if (!simplex.info.has_value()) {
        // Simplex ModelInfo should be found by now.
        // If not there's no way to figure out which stereo model to use.
        throw std::runtime_error("Cannot resolve stereo duplex model without known simplex model.");
    }

    const auto& stereo = get_stereo_model_info(simplex.info.value());
    model_sources.stereo = find_or_download_model(stereo);
}

ModelSource ModelResolver::find_or_download_model(const ModelInfo& model_info) const {
    if (m_models_directory.has_value() && !m_models_directory->empty()) {
        const auto models_dir_path = m_models_directory.value() / model_info.name;
        if (fs::exists(models_dir_path)) {
            // Model exists in models directory
            spdlog::trace(" - found model '{}' at '{}'.", model_info.name,
                          models_dir_path.string());
            return ModelSource{models_dir_path, model_info, false};
        }

        // Model not found in models directory - download it permanently
        const auto dl_model_dir_path = download(model_info, to_string(model_info.model_type));
        return ModelSource{dl_model_dir_path, model_info, false};
    }

    const auto cwd_path = fs::current_path() / model_info.name;
    if (fs::exists(cwd_path)) {
        // Model found in current working directory
        spdlog::trace(" - found model '{}' at '{}'.", model_info.name, cwd_path.string());
        return ModelSource{cwd_path, model_info, false};
    }

    // Model not found in current working directory - download it temporarily
    const auto dl_temporary_path = download(model_info, to_string(model_info.model_type));
    return ModelSource{dl_temporary_path, model_info, true};
}

void ModelResolver::model_compatibility_check(ModelSources& model_sources) const {
    const auto simplex_config = config::load_model_config(model_sources.simplex.path);
    const int simplex_sample_rate =
            simplex_config.sample_rate < 0
                    ? get_sample_rate_by_model_name(model_sources.simplex.info->name)
                    : simplex_config.sample_rate;

    int data_sample_rate = 0;
    try {
        data_sample_rate = static_cast<int>(file_info::get_sample_rate(m_reads));
    } catch (const std::exception& e) {
        spdlog::warn(
                "Could not check that model sampling rate and data sampling rate match. "
                "Proceed with caution. Reason: {}",
                e.what());
    }
    if (data_sample_rate != 0) {
        if (!utils::eq_with_tolerance(data_sample_rate, simplex_sample_rate, 100)) {
            std::string err = "Sample rate for model (" + std::to_string(simplex_sample_rate) +
                              ") and data (" + std::to_string(data_sample_rate) +
                              ") are not compatible.";
            throw std::runtime_error(err);
        }
    }
};

void ModelResolver::warn_rna_model(const ModelSource& simplex) const {
    bool is_rna{false};
    if (simplex.info.has_value()) {
        const Chemistry c = simplex.info->chemistry;
        is_rna = c == Chemistry::RNA002_70BPS || c == Chemistry::RNA004_130BPS;
    } else {
        const SampleType s = config::load_model_config(simplex.path).sample_type;
        is_rna = s == SampleType::RNA002 || s == SampleType::RNA004;
    }

    if (is_rna) {
        spdlog::info(
                " - BAM format does not support `U`, so RNA output files will include `T` "
                "instead of `U` for all file types.");
    }
};

void ModelResolver::warn_stereo_fast(const std::optional<ModelSource>& stereo) const {
    if (!stereo.has_value()) {
        return;
    }

    const auto& stereo_info = stereo.value().info;
    if (stereo_info.has_value()) {
        if (stereo_info.value().simplex.variant == ModelVariant::FAST) {
            spdlog::warn("Duplex is not supported for fast models.");
        }
    }
};

BasecallerModelResolver::BasecallerModelResolver(
        std::string model_complex,
        std::string modbase_models,
        std::vector<std::string> modbases,
        std::optional<std::string> models_directory,
        bool skip_model_compatibility_check,
        const std::vector<std::filesystem::directory_entry>& reads)
        : ModelResolver(Mode::SIMPLEX,
                        std::move(model_complex),
                        std::move(modbase_models),
                        std::move(modbases),
                        std::nullopt,
                        std::move(models_directory),
                        skip_model_compatibility_check,
                        reads) {};

DuplexModelResolver::DuplexModelResolver(std::string model_complex,
                                         std::string modbase_models,
                                         std::vector<std::string> modbases,
                                         std::optional<std::string> stereo,
                                         std::optional<std::string> models_directory,
                                         bool skip_model_compatibility_check,
                                         const std::vector<std::filesystem::directory_entry>& reads)
        : ModelResolver(Mode::DUPLEX,
                        std::move(model_complex),
                        std::move(modbase_models),
                        std::move(modbases),
                        std::move(stereo),
                        std::move(models_directory),
                        skip_model_compatibility_check,
                        reads) {};

DownloaderModelResolver::DownloaderModelResolver(
        std::string model_complex,
        std::optional<std::string> models_directory,
        const std::vector<std::filesystem::directory_entry>& reads)
        : ModelResolver(Mode::DOWNLOADER,
                        std::move(model_complex),
                        "",
                        {},
                        std::nullopt,
                        std::move(models_directory),
                        true,
                        reads) {};

models::Chemistry ModelResolver::get_chemistry() const {
    if (m_chemistry_override.has_value()) {
        return m_chemistry_override.value();
    }
    if (m_reads.empty()) {
        throw std::logic_error("No signal data to determine sequencing chemistry.");
    }
    return file_info::get_unique_sequencing_chemistry(m_reads);
};

std::filesystem::path ModelResolver::download(const models::ModelInfo& model_info,
                                              std::string_view description) const {
    if (m_download_override.has_value()) {
        return m_download_override.value()(model_info, description);
    }

    model_downloader::ModelDownloader downloader(m_models_directory, false);
    return downloader.get(model_info, description);
};

}  // namespace dorado::model_resolution