#pragma once

#include "utils/fs_utils.h"
#include "utils/string_utils.h"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace dorado::models {
struct ModelInfo;
}

namespace httplib {
class Client;
}  // namespace httplib

namespace dorado::model_downloader {

using namespace models;

bool download_models(const std::string& target_directory, const std::string& model_name);

// A ModelDownloader will download models which do not already exist in the optional `model_dir` directory.
// If `model_dir` is set then models are downloaded in to new model subdirectories there, otherwise
// models are downloaded into local temporary directories which should be cleaned
class ModelDownloader {
public:
    // Model downloader with optional model search directory
    ModelDownloader(const std::optional<std::filesystem::path>& model_dir)
            : m_models_dir(model_dir) {}

    // Temporary model downloader
    ModelDownloader() : m_models_dir(std::nullopt) {}

    // Get the model search / downloads directory if any
    std::optional<std::filesystem::path> models_directory() const { return m_models_dir; }

    // True if this downloader is downloading into a temporary directory
    bool is_temporary() const { return !m_models_dir.has_value(); }

    // Get all temporary models which should be cleaned up
    std::set<std::filesystem::path> temporary_models() const { return m_temp_models; }

    // Download a model
    std::filesystem::path get(const ModelInfo& model, const std::string& description);
    // Download multiple models
    std::vector<std::filesystem::path> get(const std::vector<ModelInfo>& models,
                                           const std::string& description);

private:
    const std::optional<std::filesystem::path> m_models_dir;
    std::set<std::filesystem::path> m_temp_models;
};

}  // namespace dorado::model_downloader
