#pragma once
#include <torch/torch.h>

namespace dorado::utils {

// Read Trimming method (removes some initial part of the raw read).
int trim(const torch::Tensor &signal,
         float threshold = 2.4,
         int window_size = 40,
         int min_elements = 3);

}  // namespace dorado::utils