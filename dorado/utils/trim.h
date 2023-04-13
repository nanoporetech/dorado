#pragma once
#include <torch/torch.h>

namespace dorado::utils {

// Read Trimming method (removes some initial part of the raw read).
int trim(torch::Tensor signal,
         int max_samples = 8000,
         float threshold = 2.4,
         int window_size = 40,
         int min_elements = 3);

}  // namespace dorado::utils