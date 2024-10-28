#pragma once

#include <filesystem>

namespace dorado::utils {

std::filesystem::path get_fai_path(const std::filesystem::path& in_fastx_fn);

bool check_fai_exists(const std::filesystem::path& in_fastx_fn);

void create_fai_index(const std::filesystem::path& in_fastx_fn);

}  // namespace dorado::utils
