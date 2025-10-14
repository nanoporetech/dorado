#pragma once

#include "model_resolver/ModelSources.h"
#include "models/kits.h"
#include "models/models.h"

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace dorado::model_resolution {

std::optional<std::filesystem::path> get_models_directory(
        const std::optional<std::string>& models_directory_arg);

class ModelResolver {
public:
    ModelSources resolve();

protected:
    enum class Mode { SIMPLEX, DUPLEX, DOWNLOADER };
    ModelResolver(Mode mode,
                  std::string model_complex_arg,
                  std::string modbase_models_arg,
                  std::vector<std::string> modbases_arg,
                  std::optional<std::string> stereo_arg,
                  std::optional<std::string> models_directory_arg,
                  bool skip_model_compatibility_check,
                  const std::vector<std::filesystem::directory_entry>& reads);

    ModelSources resolve_model_complex() const;
    void resolve_modbase_models(ModelSources& model_sources) const;
    void resolve_stereo_models(ModelSources& model_sources) const;

    void model_compatibility_check(ModelSources& model_soruces) const;
    void warn_rna_model(const ModelSource& simplex) const;
    void warn_stereo_fast(const std::optional<ModelSource>& stereo) const;

    const Mode m_mode;
    std::string m_model_complex;
    std::string m_modbase_models;
    std::vector<std::string> m_modbases;
    std::optional<std::string> m_stereo, m_models_directory_arg;
    const std::vector<std::filesystem::directory_entry>& m_reads;

    bool m_skip_model_compatibility_check{false};
    std::optional<std::filesystem::path> m_models_directory;

    // Overrides for testing purposes
    std::optional<models::Chemistry> m_chemistry_override;
    std::optional<std::function<std::filesystem::path(const models::ModelInfo& model_info,
                                                      std::string_view description)>>
            m_download_override;
    bool m_check_paths_override{true};

private:
    models::Chemistry get_chemistry() const;
    std::filesystem::path download(const models::ModelInfo& model_info,
                                   std::string_view description) const;
    ModelSource find_or_download_model(const models::ModelInfo& model_info) const;
};

class BasecallerModelResolver : public ModelResolver {
public:
    BasecallerModelResolver(std::string model_complex_arg,
                            std::string modbase_models_arg,
                            std::vector<std::string> modbases_arg,
                            std::optional<std::string> models_directory_arg,
                            bool skip_model_compatibility_check,
                            const std::vector<std::filesystem::directory_entry>& reads);
};

class DuplexModelResolver : public ModelResolver {
public:
    DuplexModelResolver(std::string model_complex_arg,
                        std::string modbase_models_arg,
                        std::vector<std::string> modbases_arg,
                        std::optional<std::string> stereo_arg,
                        std::optional<std::string> models_directory_arg,
                        bool skip_model_compatibility_check,
                        const std::vector<std::filesystem::directory_entry>& reads);
};

class DownloaderModelResolver : public ModelResolver {
public:
    DownloaderModelResolver(std::string model_complex_arg,
                            std::optional<std::string> models_directory_arg,
                            const std::vector<std::filesystem::directory_entry>& reads);
};

}  // namespace dorado::model_resolution
