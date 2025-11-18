
#include "model_resolver/Models.h"

#include "config/BatchParams.h"
#include "utils/fs_utils.h"

#include <spdlog/spdlog.h>

#include <ranges>

namespace dorado::model_resolution {

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

void Models::set_basecaller_batch_params(const config::BatchParams& batch_params,
                                         const std::string& device) {
    m_simplex_config.basecaller.update(batch_params);

    if (device == "cpu" && m_simplex_config.basecaller.batch_size() == 0) {
        // Force the batch size to 128
        // TODO: This is tuned for LSTM models - investigate Tx
        m_simplex_config.basecaller.set_batch_size(128);
    }

    m_simplex_config.normalise_basecaller_params();
};

void Models::cleanup_temporary_models() const {
    auto sources = m_sources.mods;
    sources.insert(sources.cbegin(), m_sources.simplex);
    if (m_sources.stereo.has_value()) {
        sources.push_back(m_sources.stereo.value());
    }

    const auto is_temporary = [](const ModelSource& s) { return s.is_temporary; };
    for (const auto& src : sources | std::ranges::views::filter(is_temporary)) {
        const auto parent_filename = src.path.parent_path().filename().string();
        if (parent_filename.starts_with(utils::TEMP_MODELS_DIR_PREFIX)) {
            utils::clean_temporary_models({src.path.parent_path()});
        } else {
            utils::clean_temporary_models({src.path});
        }
    }
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

}  // namespace dorado::model_resolution
