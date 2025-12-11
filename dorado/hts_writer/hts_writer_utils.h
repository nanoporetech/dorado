#pragma once

#include <spdlog/spdlog.h>

#include <filesystem>
#include <mutex>
#include <stdexcept>

namespace dorado::hts_writer {
inline void create_output_folder(const std::filesystem::path& path) {
#ifdef _WIN32
    static std::once_flag long_path_warning_flag;
    if (path.string().size() >= 260) {
        std::call_once(long_path_warning_flag, [&path] {
            spdlog::warn("Filepaths longer than 260 characters may cause issues on Windows.");
        });
    }
#endif

    if (std::filesystem::exists(path.parent_path())) {
        return;
    }

    spdlog::debug("Creating output folder: '{}'. Length:{}", path.parent_path().string(),
                  path.string().size());
    std::error_code creation_error;
    // N.B. No error code if folder already exists.
    std::filesystem::create_directories(path.parent_path(), creation_error);
    if (creation_error) {
        spdlog::error("Unable to create output folder '{}'.  ErrorCode({}) {}", path.string(),
                      creation_error.value(), creation_error.message());
        throw std::runtime_error("Failed to create output directory");
    }
}
}  // namespace dorado::hts_writer
