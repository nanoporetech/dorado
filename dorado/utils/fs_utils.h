#pragma once

#include <filesystem>
#include <optional>
#include <set>

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

}  // namespace dorado::utils