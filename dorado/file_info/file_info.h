#pragma once

#include "models/kits.h"
#include "utils/types.h"

#include <filesystem>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dorado::file_info {

/**
 * Class caching directory entries for a folder, along with the input parameters.
 */
class DirectoryFiles final {
    const std::filesystem::path m_data_path;
    bool m_recursive;
    const std::vector<std::filesystem::directory_entry> m_directory_entries;

public:
    DirectoryFiles(std::filesystem::path data_path, bool recursive);

    const std::filesystem::path& path() const;

    bool recursive() const;

    const std::vector<std::filesystem::directory_entry>& entries() const;
};

std::unordered_map<std::string, ReadGroup> load_read_groups(const DirectoryFiles& dir_files,
                                                            const std::string& model_name,
                                                            const std::string& modbase_model_names);

int get_num_reads(const DirectoryFiles& dir_files,
                  std::optional<std::unordered_set<std::string>> read_list,
                  const std::unordered_set<std::string>& ignore_read_list);

bool is_read_data_present(const DirectoryFiles& dir_files);

uint16_t get_sample_rate(const DirectoryFiles& dir_files);

// Inspects the sequencing data metadata to determine the sequencing chemistry used.
// Throws runtime_error if the data is inhomogeneous.
models::Chemistry get_unique_sequencing_chemisty(const DirectoryFiles& dir_files);

}  // namespace dorado::file_info
