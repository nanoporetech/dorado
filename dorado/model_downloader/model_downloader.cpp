#include "model_downloader.h"

#include "downloader.h"
#include "models/models.h"
#include "utils/fs_utils.h"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <vector>
namespace fs = std::filesystem;

namespace dorado::model_downloader {

bool download_models(const std::string& target_directory, const std::string& selected_model) {
    if (selected_model != "all" && !models::is_valid_model(selected_model)) {
        spdlog::error("Selected model doesn't exist: {}", selected_model);
        return false;
    }

    Downloader downloader(target_directory);

    bool success = true;
    auto download_model_set = [&](const models::ModelList& models) {
        for (const auto& model : models) {
            if (selected_model == "all" || selected_model == model.name) {
                if (!downloader.download(model)) {
                    success = false;
                }
            }
        }
    };

    download_model_set(models::simplex_models());
    download_model_set(models::stereo_models());
    download_model_set(models::modified_models());
    download_model_set(models::correction_models());
    download_model_set(models::polish_models());

    return success;
}

std::filesystem::path ModelDownloader::get(const ModelInfo& model, const std::string& description) {
    const fs::path parent_dir =
            m_models_dir.has_value() ? m_models_dir.value() : utils::create_temporary_directory();

    // clang-tidy warns about performance-no-automatic-move if |temp_model_dir| is const. It should be treated as such though.
    /*const*/ fs::path model_dir = parent_dir / model.name;

    if (fs::exists(model_dir)) {
        spdlog::trace("Found existing model at '{}'.", model_dir.u8string());
        return model_dir;
    }

    if (m_models_dir.has_value()) {
        spdlog::trace("Model does not exist at '{}' - downloading it instead.",
                      model_dir.u8string());
    }

    if (!utils::has_write_permission(parent_dir)) {
        throw std::runtime_error("Failed to prepare model download directory");
    }

    if (!download_models(parent_dir.u8string(), model.name)) {
        throw std::runtime_error("Failed to download + " + description + " model: '" + model.name +
                                 "'.");
    }

    if (is_temporary()) {
        // Check parent_dir is temp preventing unintentionally deleting work
        if (parent_dir.filename().u8string().find(utils::TEMP_MODELS_DIR_PREFIX) == 0) {
            m_temp_models.emplace(parent_dir);
        } else {
            spdlog::warn(
                    "Temporary {} model directory does not have the expected name '{}' at: '{}'",
                    description, utils::TEMP_MODELS_DIR_PREFIX, parent_dir.u8string());
        }
    }

    spdlog::trace("Downloaded {} model into: '{}'.", description, model_dir.u8string());
    return model_dir;
}

std::vector<std::filesystem::path> ModelDownloader::get(const std::vector<ModelInfo>& models,
                                                        const std::string& description) {
    std::vector<std::filesystem::path> paths;
    paths.reserve(models.size());
    for (const auto& info : models) {
        auto m = get(info, description);
        paths.push_back(m);
    }
    return paths;
}

}  // namespace dorado::model_downloader
