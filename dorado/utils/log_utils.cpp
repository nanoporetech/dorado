#include "log_utils.h"

#include "tty_utils.h"

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

#include <array>
#include <string>

namespace dorado::utils {

/**
 * @brief Get the file path associated with a given file descriptor
 *
 * This function retrieves the file path for a given file.
 * If the file descriptor does not correspond to a file, or if an error occurs, the function
 * returns an empty string.
 *
 * @param fd The file descriptor for which to retrieve the file path
 * @return A string containing the file path, or an empty string if the file path could not be retrieved
 */
#ifndef _WIN32
std::string get_file_path(int fd) {
#ifdef __APPLE__
    char filePath[1024];

    if (fcntl(fd, F_GETPATH, filePath) != -1) {
        return std::string(filePath);
    }
#endif

#ifdef __linux__
    std::array<char, 256> filePath;
    std::array<char, 256> procfdPath;
    std::snprintf(procfdPath.data(), procfdPath.size(), "/proc/self/fd/%d", fd);
    ssize_t len = readlink(procfdPath.data(), filePath.data(), filePath.size() - 1);
    if (len != -1) {
        filePath[len] = '\0';
        return std::string(filePath.data());
    }
#endif
    return "";
}
#endif  // _WIN32

bool is_safe_to_log() {
#ifdef _WIN32
    return true;
#else
    if (get_file_path(fileno(stdout)) == get_file_path(fileno(stderr))) {
        // if both stdout and stderr are ttys it's safe to log
        if (utils::is_fd_tty(stderr)) {
            return true;
        }
        return false;
    }
    return true;
#endif
}

void InitLogging() {
    // Without modification, the default logger will write to stdout.
    // Replace the default logger with a (color, multi-threaded) stderr logger
    // (but first replace it with an arbitrarily-named logger to prevent a name clash)
    spdlog::set_default_logger(spdlog::stderr_color_mt("unused_name"));
    spdlog::set_default_logger(spdlog::stderr_color_mt(""));
    if (!is_safe_to_log()) {
        spdlog::set_level(spdlog::level::off);
    }
}

void SetVerboseLogging(VerboseLogLevel level) {
    if (is_safe_to_log()) {
        if (level >= VerboseLogLevel::trace) {
            spdlog::set_level(spdlog::level::trace);
        } else if (level <= VerboseLogLevel::debug) {
            spdlog::set_level(spdlog::level::debug);
        }
    }
}

void EnsureInfoLoggingEnabled(VerboseLogLevel level) {
    switch (level) {
    case VerboseLogLevel::none:
        spdlog::set_level(spdlog::level::info);
        break;
    case VerboseLogLevel::debug:
        spdlog::set_level(spdlog::level::debug);
        break;
    case VerboseLogLevel::trace:
        spdlog::set_level(spdlog::level::trace);
        break;
    };
}

}  // namespace dorado::utils
