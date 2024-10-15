#pragma once

#include "models/kits.h"
#include "utils/types.h"

#include <filesystem>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace dorado::file_info {

std::unordered_map<std::string, ReadGroup> load_read_groups(const std::filesystem::path& data_path,
                                                            std::string model_name,
                                                            std::string modbase_model_names,
                                                            bool recursive_file_loading);

int get_num_reads(const std::filesystem::path& data_path,
                  std::optional<std::unordered_set<std::string>> read_list,
                  const std::unordered_set<std::string>& ignore_read_list,
                  bool recursive_file_loading);

bool is_read_data_present(const std::filesystem::path& data_path, bool recursive_file_loading);

uint16_t get_sample_rate(const std::filesystem::path& data_path, bool recursive_file_loading);

// Inspects the sequencing data metadata to determine the sequencing chemistry used.
// Calls get_sequencing_chemistries but will error if the data is inhomogeneous
models::Chemistry get_unique_sequencing_chemisty(const std::string& data,
                                                 bool recursive_file_loading);

std::set<models::ChemistryKey> get_sequencing_chemistries(const std::filesystem::path& data_path,
                                                          bool recursive_file_loading);

}  // namespace dorado::file_info
