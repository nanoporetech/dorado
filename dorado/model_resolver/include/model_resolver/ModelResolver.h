#pragma once

#include "config/BasecallModelConfig.h"
#include "config/ModBaseModelConfig.h"
#include "models/kits.h"
#include "models/models.h"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace dorado {

namespace model_resolution {

std::optional<std::filesystem::path> get_models_directory(
        const std::optional<std::string>& models_directory_arg);

struct ModelSource {
    std::filesystem::path path;
    std::optional<models::ModelInfo> info;
    bool is_temporary{false};

    bool operator==(const ModelSource& other) const {
        if (info.has_value() && other.info.has_value()) {
            return info.value() == other.info.value();
        };
        return path == other.path;
    };
};

struct ModelSources {
    ModelSource simplex;
    std::vector<ModelSource> mods;
    std::optional<ModelSource> stereo;

    bool operator==(const ModelSources& other) const {
        if (!(simplex == other.simplex)) {
            return false;
        }

        if (mods.size() != other.mods.size()) {
            return false;
        }

        for (const auto& m : mods) {
            bool found_match = false;
            for (const auto& o : other.mods) {
                if (m == o) {
                    found_match = true;
                }
            }
            if (!found_match) {
                return false;
            }
        }

        if (stereo.has_value() && other.stereo.has_value()) {
            if (!(stereo.value() == other.stereo.value())) {
                return false;
            }
        }

        return true;
    };

    bool operator!=(const ModelSources& other) const { return !(*this == other); }

    bool check_paths() const;
};

class Models {
public:
    Models(const ModelSources& sources);
    Models(const Models&) = delete;

    ~Models() { cleanup_temporary_models(); };

    const ModelSources& get_sources() const { return m_sources; }

    const config::BasecallModelConfig& get_simplex_config() const;
    const std::vector<config::ModBaseModelConfig>& get_modbase_configs() const;

    std::string get_simplex_model_name() const;

    std::vector<std::string> get_modbase_model_names() const;

    std::filesystem::path get_simplex_model_path() const;
    std::vector<std::filesystem::path> get_modbase_model_paths() const;

    void set_basecaller_batch_params(const config::BatchParams& batch_params,
                                     const std::string& device);

    bool operator==(const Models& other) const { return get_sources() == other.get_sources(); }

    void print(const std::string& context) const;

protected:
    const ModelSources m_sources;

    config::BasecallModelConfig m_simplex_config;
    std::vector<config::ModBaseModelConfig> m_modbase_configs;

    void cleanup_temporary_models() const;
};

class DuplexModels final : public Models {
public:
    DuplexModels(const ModelSources& sources);
    DuplexModels(const DuplexModels&) = delete;

    std::string get_stereo_model_name() const;
    std::filesystem::path get_stereo_model_path() const;
    const config::BasecallModelConfig& get_stereo_config() const;

    void set_basecaller_batch_params(const config::BatchParams& batch_params,
                                     const std::string& device);

    void print(const std::string& context) const;

private:
    config::BasecallModelConfig m_stereo_config;
};

class ModelResolver {
public:
    ModelSources resolve();

protected:
    enum class Mode { SIMPLEX, DUPLEX, DOWNLOADER };
    ModelResolver(Mode mode,
                  const std::string& model_complex_arg,
                  const std::string& modbase_models_arg,
                  const std::vector<std::string>& modbases_arg,
                  const std::optional<std::string>& stereo_arg,
                  const std::optional<std::string>& models_directory_arg,
                  bool skip_model_compatibility_check,
                  const std::vector<std::filesystem::directory_entry>& reads);

    ModelSources resolve_model_complex() const;
    void resolve_modbase_models(ModelSources& model_sources) const;
    void resolve_stereo_models(ModelSources& model_sources) const;

    void model_compatibility_check(ModelSources& model_soruces) const;
    void warn_rna_model(const ModelSource& simplex) const;
    void warn_stereo_fast(const std::optional<ModelSource>& stereo) const;

    const Mode m_mode;
    std::string m_model_complex_arg;
    std::string m_modbase_models_arg;
    std::vector<std::string> m_modbases_arg;
    std::optional<std::string> m_stereo_arg, m_models_directory_arg;
    std::vector<std::filesystem::directory_entry> m_reads;

    bool m_skip_model_compatibility_check{false};
    std::optional<std::filesystem::path> m_models_directory;

    // Overrides for testing purposes
    std::optional<models::Chemistry> m_chemistry_override;
    std::optional<std::function<std::filesystem::path(const models::ModelInfo& model_info,
                                                      const std::string& description)>>
            m_download_override;
    bool m_check_paths_override{true};

private:
    models::Chemistry get_chemistry() const;
    std::filesystem::path download(const models::ModelInfo& model_info,
                                   const std::string& description) const;
    ModelSource find_or_download_model(const models::ModelInfo& model_info) const;
};

class BasecallerModelResolver : public ModelResolver {
public:
    BasecallerModelResolver(const std::string& model_complex_arg,
                            const std::string& modbase_models_arg,
                            const std::vector<std::string>& modbases_arg,
                            const std::optional<std::string>& models_directory_arg,
                            bool skip_model_compatibility_check,
                            const std::vector<std::filesystem::directory_entry>& reads);
};

class DuplexModelResolver : public ModelResolver {
public:
    DuplexModelResolver(const std::string& model_complex_arg,
                        const std::string& modbase_models_arg,
                        const std::vector<std::string>& modbases_arg,
                        const std::optional<std::string>& stereo_arg,
                        const std::optional<std::string>& models_directory_arg,
                        bool skip_model_compatibility_check,
                        const std::vector<std::filesystem::directory_entry>& reads);
};

class DownloaderModelResolver : public ModelResolver {
public:
    DownloaderModelResolver(const std::string& model_complex_arg,
                            const std::optional<std::string>& models_directory_arg,
                            const std::vector<std::filesystem::directory_entry>& reads);
};

}  // namespace model_resolution

}  // namespace dorado