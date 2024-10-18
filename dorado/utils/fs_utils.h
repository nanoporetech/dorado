#pragma once

#include <filesystem>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace dorado::utils {

const std::string TEMP_MODELS_DIR_PREFIX{".temp_dorado_model-"};

// True if the caller has permission to write to files in directory. If directory
// does not exist, then it is created. Exceptions are discarded but error messages are issued.
bool has_write_permission(const std::filesystem::path& directory);

// Returns a randomly generated filepath in the current working directory (cross-platform)
std::filesystem::path create_temporary_directory();

// Returns the a temporary directory or the path provided by override after asserting
// write permissions. Throws runtime_error otherwise.
std::filesystem::path get_downloads_path(const std::optional<std::filesystem::path>& override);

// Removes paths
void clean_temporary_models(const std::set<std::filesystem::path>& paths);

/**
 * @brief Fetches directory entries from a specified path.
 *
 * This function fetches all non-folder directory entries from the specified path. If the path is not a 
 * directory, it will return a vector containing a single entry representing the specified file.
 * It can operate in two modes: recursive and non-recursive. In recursive mode, it fetches entries from
 * all subdirectories recursively. In non-recursive mode, it only fetches entries from the top-level directory.
 *
 * @param path The path from which to fetch the directory entries. It can be a path to a file or a directory.
 * @param recursive A boolean flag indicating whether to operate in recursive mode.
 *                  True for recursive mode, false for non-recursive mode.
 * @return A vector of directory entries fetched from the specified path.
 */
std::vector<std::filesystem::directory_entry> fetch_directory_entries(
        const std::filesystem::path& path,
        bool recursive);

}  // namespace dorado::utils