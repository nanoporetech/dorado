#pragma once

#include "models/kits.h"
#include "utils/fs_utils.h"
#include "utils/types.h"

#include <filesystem>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dorado::file_info {

std::unordered_map<std::string, ReadGroup> load_read_groups(const utils::DirectoryFiles& dir_files,
                                                            const std::string& model_name,
                                                            const std::string& modbase_model_names);

int get_num_reads(const utils::DirectoryFiles& dir_files,
                  std::optional<std::unordered_set<std::string>> read_list,
                  const std::unordered_set<std::string>& ignore_read_list);

bool is_read_data_present(const utils::DirectoryFiles& dir_files);

uint16_t get_sample_rate(const utils::DirectoryFiles& dir_files);

// Inspects the sequencing data metadata to determine the sequencing chemistry used.
// Throws runtime_error if the data is inhomogeneous.
models::Chemistry get_unique_sequencing_chemisty(const utils::DirectoryFiles& dir_files);

}  // namespace dorado::file_info
