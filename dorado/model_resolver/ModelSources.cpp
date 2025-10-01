
#include "model_resolver/ModelSources.h"

#include <spdlog/spdlog.h>

namespace dorado {

namespace model_resolution {
namespace fs = std::filesystem;

bool check_model_path(const fs::path& model_path, bool verbose) {
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

bool ModelSource::operator==(const ModelSource& other) const {
    if (info.has_value() && other.info.has_value()) {
        return info.value() == other.info.value();
    };
    return path == other.path;
};
bool ModelSources::operator==(const ModelSources& other) const {
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
}  // namespace model_resolution
}  // namespace dorado