#pragma once

#include "config/BasecallModelConfig.h"
#include "config/ModBaseModelConfig.h"
#include "model_resolver/ModelSources.h"

#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace dorado::model_resolution {

class Models {
public:
    explicit Models(ModelSources sources);

    ~Models() { cleanup_temporary_models(); };

    const ModelSources& get_sources() const { return m_sources; }

    const config::BasecallModelConfig& get_simplex_config() const;
    std::span<const config::ModBaseModelConfig> get_modbase_configs() const;

    std::string get_simplex_model_name() const;
    std::filesystem::path get_simplex_model_path() const;

    std::vector<std::string> get_modbase_model_names() const;
    std::vector<std::filesystem::path> get_modbase_model_paths() const;

    void set_basecaller_batch_params(const config::BatchParams& batch_params,
                                     const std::string& device);

    bool operator==(const Models& other) const { return get_sources() == other.get_sources(); }

    void print(const std::string& context) const;

private:
    Models(const Models&) = delete;
    Models& operator=(const Models&) = delete;
    Models(Models&&) = delete;
    Models& operator=(Models&&) = delete;

protected:
    const ModelSources m_sources;

    config::BasecallModelConfig m_simplex_config;
    std::vector<config::ModBaseModelConfig> m_modbase_configs;

    void cleanup_temporary_models() const;
};

class DuplexModels final : public Models {
public:
    explicit DuplexModels(const ModelSources& sources);

    std::string get_stereo_model_name() const;
    std::filesystem::path get_stereo_model_path() const;
    const config::BasecallModelConfig& get_stereo_config() const;

    void set_basecaller_batch_params(const config::BatchParams& batch_params,
                                     const std::string& device);

    void print(const std::string& context) const;

private:
    config::BasecallModelConfig m_stereo_config;
};

}  // namespace dorado::model_resolution