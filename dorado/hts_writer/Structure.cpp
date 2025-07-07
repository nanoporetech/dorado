#include "hts_writer/Structure.h"

#include <spdlog/spdlog.h>

#include <memory>

namespace dorado {

namespace hts_writer {

namespace fs = std::filesystem;

namespace {
std::tm get_gmtime(const std::time_t* time) {
    // gmtime is not threadsafe, so lock.
    static std::mutex gmtime_mutex;
    std::lock_guard lock(gmtime_mutex);
    std::tm* time_buffer = gmtime(time);
    return *time_buffer;
}
}  // namespace

void SingleFileStructure::create_output_folder() const {
    std::error_code creation_error;
    // N.B. No error code if folder already exists.
    const auto parent = fs::path(*m_path).parent_path();
    fs::create_directories(parent, creation_error);
    if (creation_error) {
        spdlog::error("Unable to create output folder '{}'.  ErrorCode({}) {}", parent.string(),
                      creation_error.value(), creation_error.message());
        throw std::runtime_error("Failed to create output directory");
    }
}

std::shared_ptr<const std::string> SingleFileStructure::make_shared_path(
        const std::string& output_dir) const {
    const auto path = fs::path(output_dir) / get_filename();
    return std::make_shared<const std::string>(path.string());
}

constexpr std::string_view OUTPUT_FILE_PREFIX{"calls_"};

std::string SingleFileStructure::get_filename() const {
    time_t time_now = time(nullptr);
    std::tm gm_time_now = get_gmtime(&time_now);
    char timestamp_buffer[32];
    strftime(timestamp_buffer, 32, "%F_T%H-%M-%S", &gm_time_now);

    std::ostringstream oss{};
    oss << OUTPUT_FILE_PREFIX << timestamp_buffer << get_suffix(m_mode);
    return oss.str();
}

}  // namespace hts_writer
}  // namespace dorado