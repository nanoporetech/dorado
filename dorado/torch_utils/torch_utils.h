#pragma once

namespace dorado::utils {

void initialise_torch();
void make_torch_deterministic();
void set_torch_allocator_max_split_size();

}  // namespace dorado::utils
