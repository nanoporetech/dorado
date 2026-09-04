#pragma once

#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace dorado::tests {

std::filesystem::path get_data_dir(const std::string& sub_dir);

// Reads into a string.
std::string ReadFileIntoString(const std::filesystem::path& path);

// Reads into a vector<uint8_t>.
std::vector<uint8_t> ReadFileIntoVector(const std::filesystem::path& path);

inline auto get_fast5_data_dir() { return get_data_dir("fast5"); }
inline auto get_pod5_data_dir() { return get_data_dir("pod5"); }
inline auto get_single_pod5_file_path() { return (get_data_dir("pod5") / "single_na24385.pod5"); }
inline auto get_nested_pod5_data_dir() { return get_data_dir("nested_pod5_folder"); }
inline auto get_stereo_data_dir() { return get_data_dir("stereo"); }
inline auto get_aligner_data_dir() { return get_data_dir("aligner_test"); }

// Wrapper around a temporary directory since one doesn't exist in the standard
struct TempDir {
private:
    // Force devs to use the function instead of creating their own TempDirs
    friend TempDir make_temp_dir(const std::string& prefix);
    explicit TempDir(std::filesystem::path path) : m_path(std::move(path)) {}

public:
    ~TempDir();

    TempDir(TempDir&& other) noexcept { std::swap(m_path, other.m_path); }
    TempDir& operator=(TempDir&& other) = delete;

    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    std::filesystem::path m_path;
};

TempDir make_temp_dir(const std::string& prefix);

std::string generate_random_sequence_string(int len);

}  // namespace dorado::tests

using namespace dorado::tests;
