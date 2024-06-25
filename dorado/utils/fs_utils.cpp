
#include "fs_utils.h"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <fstream>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace dorado::utils {

bool has_write_permission(const fs::path& directory) {
    if (!fs::exists(directory)) {
        try {
            fs::create_directories(directory);
        } catch (const fs::filesystem_error& e) {
            spdlog::error("{}", e.code().message());
            return false;
        }
    }

    const std::string fname = "tmp-write-test";
    std::ofstream tmp(directory / fname);
    tmp << "test";
    tmp.close();

    if (tmp.fail()) {
        spdlog::error("Insufficient permissions to download models into {}", directory.u8string());
        return false;
    }

    try {
        fs::remove(directory / fname);
    } catch (const fs::filesystem_error& e) {
        spdlog::error("{}", e.code().message());
        return false;
    }
    return true;
}

fs::path create_temporary_directory() {
    auto cwd = fs::current_path();

    std::random_device device;
    std::mt19937 rng(device());
    std::uniform_int_distribution<unsigned long long> rand(0);

    fs::path path;
    const uint16_t max_attempts = 1000;
    int16_t attempt = 0;
    while (attempt < max_attempts) {
        attempt++;

        std::stringstream ss;
        ss << std::hex << rand(rng);
        path = cwd / (utils::TEMP_MODELS_DIR_PREFIX + ss.str());
        if (fs::create_directory(path)) {
            return path;
        }
    }
    throw std::runtime_error("Failed to create temporary directory");
}

fs::path get_downloads_path(const std::optional<fs::path>& override) {
    fs::path path = override.has_value() ? override.value() : create_temporary_directory();
    if (!has_write_permission(path)) {
        throw std::runtime_error("Failed to prepare model download directory");
    }
    return path;
}

void clean_temporary_models(const std::set<std::filesystem::path>& paths) {
    for (const auto& path : paths) {
        spdlog::trace("Deleting temporary model path: {}", path.u8string());
        try {
            fs::remove_all(path);
        } catch (const fs::filesystem_error& e) {
            spdlog::trace("Failed to clean temporary model path - {}", e.what());
        }
    }
}

}  // namespace dorado::utils