#pragma once

#include <torch/torch.h>

#include <filesystem>
#include <string>
#include <vector>

namespace utils {

// Serialise Torch tensor to disk.
void serialise_tensor(torch::Tensor t, const std::string& path);
// Load serialised tensor from disk.
std::vector<torch::Tensor> load_tensors(const std::filesystem::path& dir,
                                        const std::vector<std::string>& tensors);

torch::Tensor quantile(const torch::Tensor t, const torch::Tensor q);
torch::Tensor quantile_counting(const torch::Tensor t,
                                int range_min,
                                int range_max,
                                const torch::Tensor q);

}  // namespace utils
