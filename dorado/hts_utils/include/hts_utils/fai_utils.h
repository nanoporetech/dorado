#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace dorado::utils {

std::filesystem::path get_fai_path(const std::filesystem::path& in_fastx_fn);

bool check_fai_exists(const std::filesystem::path& in_fastx_fn);

bool create_fai_index(const std::filesystem::path& in_fastx_fn);

std::vector<std::pair<std::string, int64_t>> load_seq_lengths(
        const std::filesystem::path& in_fastx_fn);

}  // namespace dorado::utils
