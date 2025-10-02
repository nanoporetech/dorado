#pragma once

#include "models/models.h"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <optional>
#include <vector>

namespace dorado {

namespace model_resolution {

bool check_model_path(const std::filesystem::path& model_path, bool verbose);

struct ModelSource {
    std::filesystem::path path;
    std::optional<models::ModelInfo> info;
    bool is_temporary{false};

    bool operator==(const ModelSource& other) const;
};

struct ModelSources {
    ModelSource simplex;
    std::vector<ModelSource> mods;
    std::optional<ModelSource> stereo;

    bool operator==(const ModelSources& other) const;
    bool operator!=(const ModelSources& other) const { return !(*this == other); }

    bool check_paths() const;
};

std::string to_string(const ModelSource& ms);
std::string to_string(const ModelSources& mss);

std::ostream& operator<<(std::ostream& oss, const ModelSource& ms);
std::ostream& operator<<(std::ostream& oss, const ModelSources& mss);

}  // namespace model_resolution
}  // namespace dorado