#include "model_resolver/ModelSources.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <sstream>

namespace dorado::model_resolution {
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

    if (!std::ranges::is_permutation(mods, other.mods)) {
        return false;
    }

    if (stereo.has_value() && other.stereo.has_value()) {
        if (!(stereo.value() == other.stereo.value())) {
            return false;
        }
    }

    return true;
};

std::ostream& operator<<(std::ostream& oss, const ModelSource& ms) {
    oss << "ModelSource{path='" << ms.path.string() << "', info=";
    oss << (ms.info.has_value() ? "known" : "unknown") << ", temporary=";
    oss << (ms.is_temporary ? "true" : "false") << "}";
    return oss;
}

std::string to_string(const ModelSource& ms) {
    std::ostringstream oss;
    oss << ms;
    return oss.str();
}

std::ostream& operator<<(std::ostream& oss, const ModelSources& mss) {
    oss << "ModelSources{simplex=" << mss.simplex << ", mods=[";
    for (std::size_t i = 0; i < mss.mods.size(); ++i) {
        if (i) {
            oss << ", ";
        }
        oss << mss.mods[i];
    }
    oss << "], stereo=";
    if (mss.stereo.has_value()) {
        oss << *mss.stereo;
    } else {
        oss << "none";
    }
    oss << "}";
    return oss;
}

std::string to_string(const ModelSources& mss) {
    std::ostringstream oss;
    oss << mss;
    return oss.str();
}

}  // namespace dorado::model_resolution