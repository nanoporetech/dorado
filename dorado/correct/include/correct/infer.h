#pragma once

#include "torch_utils/gpu_profiling.h"
#include "types.h"

#include <spdlog/spdlog.h>
#include <torch/torch.h>

#include <filesystem>

#ifdef NDEBUG
#define LOG_TRACE(...)
#else
#define LOG_TRACE(...) spdlog::trace(__VA_ARGS__)
#endif

namespace dorado::correction {

// Custom collate function. Replacement for torch::utils::rnn::pad_sequence
// because that was running much slower than this version.
template <typename T>
torch::Tensor collate(std::vector<torch::Tensor>& tensors,
                      T fill_val,
                      torch::ScalarType type,
                      const bool pinned_memory) {
    dorado::utils::ScopedProfileRange spr("collate", 1);
    auto max_length = std::max_element(tensors.begin(), tensors.end(),
                                       [](const torch::Tensor& a, const torch::Tensor& b) {
                                           return a.sizes()[0] < b.sizes()[0];
                                       })
                              ->sizes()[0];
    auto max_reads = std::max_element(tensors.begin(), tensors.end(),
                                      [](const torch::Tensor& a, const torch::Tensor& b) {
                                          return a.sizes()[1] < b.sizes()[1];
                                      })
                             ->sizes()[1];

    auto options =
            torch::TensorOptions().dtype(type).device(torch::kCPU).pinned_memory(pinned_memory);
    torch::Tensor batch = torch::empty({(int)tensors.size(), max_length, max_reads}, options);

    T* ptr = batch.data_ptr<T>();
    std::fill(ptr, ptr + batch.numel(), fill_val);

    // Copy over data for each tensor
    for (size_t i = 0; i < tensors.size(); i++) {
        torch::Tensor slice = batch.index({(int)i, torch::indexing::Slice(0, tensors[i].sizes()[0]),
                                           torch::indexing::Slice(0, tensors[i].sizes()[1])});
        slice.copy_(tensors[i]);
    }

    LOG_TRACE("size {}x{}x{} numelem {} sum {}", tensors.size(), max_length, max_reads,
              batch.numel(), batch.sum().item<T>());

    return batch;
}

int calculate_batch_size(const std::string& device, float memory_fraction);

ModelConfig parse_model_config(const std::filesystem::path& config_path);

}  // namespace dorado::correction
