#pragma once

#include <string>

namespace dorado::utils {

void initialise_torch();
void make_torch_deterministic();
void set_torch_allocator_max_split_size();

std::string torch_stacktrace();

}  // namespace dorado::utils
