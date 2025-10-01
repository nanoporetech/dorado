#include "model_resolver/ModelResolver.h"

#include "config/BasecallModelConfig.h"
#include "config/BatchParams.h"
#include "config/ModBaseModelConfig.h"
#include "file_info/file_info.h"
#include "model_downloader/model_downloader.h"
#include "models/kits.h"
#include "models/metadata.h"
#include "models/model_complex.h"
#include "models/models.h"
#include "utils/fs_utils.h"
#include "utils/math_utils.h"
#include "utils/string_utils.h"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace dorado {

namespace model_resolution {
using namespace models;
namespace fs = std::filesystem;

namespace {

ModelComplex parse_model_complex(const std::string& model_arg) {
    try {
        return ModelComplex::parse(model_arg);
    } catch (std::exception& e) {
        spdlog::error("Failed to parse model argument '{}'. '{}'", model_arg, e.what());
        std::exit(EXIT_FAILURE);
    }
}

bool check_model_path(const fs::path& model_path, bool verbose) noexcept {
    try {
        const auto& p = fs::weakly_canonical(model_path);
        if (!fs::exists(p)) {
            if (verbose) {
                spdlog::error(
                        "Model does not exist at: '{}' - Please download the model or use a model "
                        "complex to automatically download a model",
                        p.string());
            }
            return false;
        }
        if (!fs::is_directory(p)) {
            if (verbose) {
                spdlog::error(
                        "Model is not a directory at: '{}' - Please check your model argument.",
                        p.string());
            }
            return false;
        }
        if (fs::is_empty(p)) {
            if (verbose) {
                spdlog::error(
                        "Model is an empty directory at: '{}' - Please check your model path.",
                        p.string());
            }
            return false;
        }
        const auto cfg = p / "config.toml";
        if (!fs::exists(cfg)) {
            if (verbose) {
                spdlog::error(
                        "Model directory is missing a configuration file at: '{}' - Please check "
                        "your model path or download your model again.",
                        cfg.string());
            }
            return false;
        }
    } catch (std::exception& e) {
        spdlog::error("Exception while checking model path at: '{}' - {}", model_path.string(),
                      e.what());
        return false;
    }
    return true;
}

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

bool ModelSources::check_paths() const {
    bool ok = true;
    ok &= check_model_path(simplex.path, true);
    for (const auto& mod : mods) {
        ok &= check_model_path(mod.path, true);
    }
    if (stereo.has_value()) {
        ok &= check_model_path(stereo->path, true);
    }
    return ok;
};

// Get the model search directory with the command line argument taking priority over the environment variable.
// Returns std:nullopt if nether are set explicitly
std::optional<fs::path> get_models_directory(
        const std::optional<std::string>& models_directory_arg) {
    const char* env_path = std::getenv("DORADO_MODELS_DIRECTORY");

    if (models_directory_arg.has_value()) {
        auto path = fs::path(models_directory_arg.value());
        if (!fs::exists(path)) {
            spdlog::error("--models-directory path does not exist at: '{}'.", path.string());
            std::exit(EXIT_FAILURE);
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

Models::Models(const ModelSources& sources) : m_sources(sources) {
    m_simplex_config = config::load_model_config(m_sources.simplex.path);

    for (const auto& mod : m_sources.mods) {
        m_modbase_configs.push_back(config::load_modbase_model_config(mod.path));
    }
};

const config::BasecallModelConfig& Models::get_simplex_config() const { return m_simplex_config; };

const std::vector<config::ModBaseModelConfig>& Models::get_modbase_configs() const {
    return m_modbase_configs;
};

std::string Models::get_simplex_model_name() const {
    return m_sources.simplex.path.filename().string();
};

std::vector<std::string> Models::get_modbase_model_names() const {
    std::vector<std::string> names;
    names.reserve(m_sources.mods.size());
    for (const auto& mod : m_sources.mods) {
        names.push_back(mod.path.filename().string());
    }
    return names;
};

std::filesystem::path Models::get_simplex_model_path() const { return m_sources.simplex.path; };

std::vector<std::filesystem::path> Models::get_modbase_model_paths() const {
    std::vector<std::filesystem::path> paths;
    paths.reserve(m_sources.mods.size());
    for (const auto& mod : m_sources.mods) {
        paths.push_back(mod.path);
    }
    return paths;
};

DuplexModels::DuplexModels(const ModelSources& sources) : Models(sources) {
    if (!m_sources.stereo.has_value()) {
        throw std::logic_error("Missing stereo model source.");
    }
    m_stereo_config = config::load_model_config(m_sources.stereo->path);
};

std::string DuplexModels::get_stereo_model_name() const {
    return m_sources.stereo->path.filename().string();
};

const config::BasecallModelConfig& DuplexModels::get_stereo_config() const {
    return m_stereo_config;
};

std::filesystem::path DuplexModels::get_stereo_model_path() const {
    return m_sources.stereo->path;
};

ModelResolver::ModelResolver(Mode mode,
                             const std::string& model_complex_arg,
                             const std::string& modbase_models_arg,
                             const std::vector<std::string>& modbases_arg,
                             const std::optional<std::string>& stereo_arg,
                             const std::optional<std::string>& models_directory_arg,
                             bool skip_model_compatibility_check,
                             const std::vector<std::filesystem::directory_entry>& reads)
        : m_mode(mode),
          m_model_complex_arg(model_complex_arg),
          m_modbase_models_arg(modbase_models_arg),
          m_modbases_arg(modbases_arg),
          m_stereo_arg(stereo_arg),
          m_models_directory_arg(models_directory_arg),
          m_reads(reads),
          m_skip_model_compatibility_check(skip_model_compatibility_check) {}

ModelSources ModelResolver::resolve() {
    m_models_directory = get_models_directory(m_models_directory_arg);

    if (m_model_complex_arg.empty()) {
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
    const auto model_complex = parse_model_complex(m_model_complex_arg);

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

    throw std::logic_error("Failed to resolve Model Complex: '" + m_model_complex_arg + "'.");
}

void ModelResolver::resolve_modbase_models(ModelSources& model_sources) const {
    const bool used_complex = !model_sources.mods.empty();
    const bool used_paths = !m_modbase_models_arg.empty();
    const bool used_modbases = !m_modbases_arg.empty();

    if (used_complex && (used_paths || used_modbases)) {
        throw std::logic_error(
                "Modbase models set via model complex, --modified-bases and "
                "--modified-bases-models are mutually exclusive.");
    }

    const auto& simplex = model_sources.simplex;

    if (used_modbases) {
        model_sources.mods.reserve(m_modbases_arg.size());
        for (const auto& modbase_arg : m_modbases_arg) {
            const auto modbase_varint = get_mods_variant(modbase_arg);
            const auto simplex_name = simplex.info.has_value() ? simplex.info->name
                                                               : simplex.path.filename().string();
            const auto mods_info = get_modification_model(
                    simplex_name, ModsVariantPair{modbase_varint, ModelVersion::NONE});

            if (simplex.info.has_value() && (simplex.info->simplex != mods_info.simplex)) {
                spdlog::error("Modbases model '{}' is incompatible with simplex model: '{}'",
                              mods_info.name, simplex.info->name);
                throw std::runtime_error("Incompatible simplex and modbases models.");
            }

            for (const auto& prev : model_sources.mods) {
                if (prev.info->name == mods_info.name) {
                    spdlog::error("Duplicate modbases model found: '{}'", prev.info->name);
                    throw std::runtime_error("Invalid modbases models arguments.");
                }
            }

            const auto mods_source = find_or_download_model(mods_info);
            model_sources.mods.push_back(mods_source);
        }
    }

    if (used_paths) {
        const auto splits = utils::split(m_modbase_models_arg, ',');
        model_sources.mods.reserve(splits.size());
        for (const auto& part : splits) {
            const auto path = fs::path(part);
            const auto maybe_info = get_model_info_from_path(path);

            for (const auto& prev : model_sources.mods) {
                if (prev.path.filename().string() == path.filename().string()) {
                    spdlog::error("Duplicate modbases model found: '{}'", path.filename().string());
                    throw std::runtime_error("Invalid modbases models arguments.");
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

    if (m_stereo_arg.has_value() && !m_stereo_arg->empty()) {
        // Use the stero model path provided
        const auto path = fs::path(m_stereo_arg.value());
        if (!fs::exists(path)) {
            spdlog::error("--stereo-model does not exist at: '{}'.", path.string());
            throw std::runtime_error("Invalud stereo model.");
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
    // TODO: Review decision to prioritise models-directory
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

void ModelResolver::model_compatibility_check(ModelSources& model_soruces) const {
    const auto simplex_config = config::load_model_config(model_soruces.simplex.path);
    const int simplex_sample_rate =
            simplex_config.sample_rate < 0
                    ? get_sample_rate_by_model_name(model_soruces.simplex.info->name)
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
        switch (simplex.info->chemistry) {
        case models::Chemistry::RNA002_70BPS:
        case models::Chemistry::RNA004_130BPS:
            is_rna = true;
            break;
        case models::Chemistry::UNKNOWN:
            throw std::logic_error("Unknown chemistry.");
        default:
            return;
        }
    } else {
        switch (config::load_model_config(simplex.path).sample_type) {
        case models::SampleType::RNA002:
        case models::SampleType::RNA004:
            is_rna = true;
            break;
        case models::SampleType::UNKNOWN:
            throw std::logic_error("Unknown sample_type.");
        default:
            return;
        };
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
        const std::string& model_complex_arg,
        const std::string& modbase_models_arg,
        const std::vector<std::string>& modbases_arg,
        const std::optional<std::string>& models_directory_arg,
        bool skip_model_compatibility_check,
        const std::vector<std::filesystem::directory_entry>& reads)
        : ModelResolver(Mode::SIMPLEX,
                        model_complex_arg,
                        modbase_models_arg,
                        modbases_arg,
                        std::nullopt,
                        models_directory_arg,
                        skip_model_compatibility_check,
                        reads) {};

DuplexModelResolver::DuplexModelResolver(const std::string& model_complex_arg,
                                         const std::string& modbase_models_arg,
                                         const std::vector<std::string>& modbases_arg,
                                         const std::optional<std::string>& stereo_arg,
                                         const std::optional<std::string>& models_directory_arg,
                                         bool skip_model_compatibility_check,
                                         const std::vector<std::filesystem::directory_entry>& reads)
        : ModelResolver(Mode::DUPLEX,
                        model_complex_arg,
                        modbase_models_arg,
                        modbases_arg,
                        stereo_arg,
                        models_directory_arg,
                        skip_model_compatibility_check,
                        reads) {};

DownloaderModelResolver::DownloaderModelResolver(
        const std::string& model_complex_arg,
        const std::optional<std::string>& models_directory_arg,
        const std::vector<std::filesystem::directory_entry>& reads)
        : ModelResolver(Mode::DOWNLOADER,
                        model_complex_arg,
                        "",
                        {},
                        std::nullopt,
                        models_directory_arg,
                        true,
                        reads) {};

void Models::set_basecaller_batch_params(const config::BatchParams& batch_params,
                                         const std::string& device) {
    m_simplex_config.basecaller.update(batch_params);
    m_simplex_config.normalise_basecaller_params();

    if (device == "cpu" && m_simplex_config.basecaller.batch_size() == 0) {
        // Force the batch size to 128
        // TODO: This is tuned for LSTM models - investigate Tx
        m_simplex_config.basecaller.set_batch_size(128);
    }
};

void Models::cleanup_temporary_models() const {
    auto sources = m_sources.mods;
    sources.insert(sources.cbegin(), m_sources.simplex);
    if (m_sources.stereo.has_value()) {
        sources.push_back(m_sources.stereo.value());
    }

    const auto is_temporary = [](const ModelSource& s) { return s.is_temporary; };
    for (const auto& src : sources | std::ranges::views::filter(is_temporary)) {
        utils::clean_temporary_models({src.path});
    }
};

void DuplexModels::set_basecaller_batch_params(const config::BatchParams& batch_params,
                                               const std::string& device) {
    Models::set_basecaller_batch_params(batch_params, device);

    auto& stereo_config = m_stereo_config;
    stereo_config.basecaller.update(batch_params);
    stereo_config.normalise_basecaller_params();

#if DORADO_METAL_BUILD
    if (device == "metal" && stereo_config.is_lstm_model()) {
        // ALWAYS auto tune the duplex batch size (i.e. batch_size passed in is 0.)
        // EXCEPT for on metal
        // For now, the minimal batch size is used for the duplex model.
        stereo_config.basecaller.set_batch_size(48);
    }
#endif
    if (device == "cpu" && stereo_config.basecaller.batch_size() == 0) {
        stereo_config.basecaller.set_batch_size(128);
    }
};

void Models::print(const std::string& context) const {
    spdlog::info("{} simplex model: '{}'", context, get_simplex_model_name());
    for (const auto& mod : get_modbase_model_names()) {
        spdlog::info("{} modbase model: '{}'", context, mod);
    }
};

void DuplexModels::print(const std::string& context) const {
    Models::print(context);
    spdlog::info("{} stereo model : '{}'", context, get_stereo_model_name());
};

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
                                              const std::string& description) const {
    if (m_download_override.has_value()) {
        return m_download_override.value()(model_info, description);
    }

    model_downloader::ModelDownloader downloader(m_models_directory, false);
    return downloader.get(model_info, description);
};

}  // namespace model_resolution
}  // namespace dorado