#pragma once

#include <torch/torch.h>

#include <string>
#include <vector>

namespace utils {

// Serialise Torch tensor to disk.
void serialise_tensor(torch::Tensor t, const std::string& path);
// Load serialised tensor from disk.
std::vector<torch::Tensor> load_tensors(const std::string& dir,
                                        const std::vector<std::string>& tensors);

}  // namespace utils
