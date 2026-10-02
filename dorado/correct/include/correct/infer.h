#pragma once

#include "torch_utils/gpu_profiling.h"
#include "types.h"
#include "utils/log_utils.h"

#include <ATen/TensorIndexing.h>
#include <ATen/ops/empty.h>

#include <filesystem>

namespace dorado::correction {

// Custom collate function. Replacement for torch::utils::rnn::pad_sequence
// because that was running much slower than this version.
template <typename T>
at::Tensor collate(std::vector<at::Tensor>& tensors,
                   T fill_val,
                   at::ScalarType type,
                   const bool pinned_memory) {
    dorado::utils::ScopedProfileRange spr("collate", 1);
    auto max_length = std::max_element(tensors.begin(), tensors.end(),
                                       [](const at::Tensor& a, const at::Tensor& b) {
                                           return a.sizes()[0] < b.sizes()[0];
                                       })
                              ->sizes()[0];
    auto max_reads = std::max_element(tensors.begin(), tensors.end(),
                                      [](const at::Tensor& a, const at::Tensor& b) {
                                          return a.sizes()[1] < b.sizes()[1];
                                      })
                             ->sizes()[1];

    auto options = at::TensorOptions().dtype(type).device(at::kCPU).pinned_memory(pinned_memory);
    at::Tensor batch = at::empty({(int)tensors.size(), max_length, max_reads}, options);

    T* ptr = batch.data_ptr<T>();
    std::fill(ptr, ptr + batch.numel(), fill_val);

    // Copy over data for each tensor
    for (size_t i = 0; i < tensors.size(); i++) {
        at::Tensor slice = batch.index({(int)i, at::indexing::Slice(0, tensors[i].sizes()[0]),
                                        at::indexing::Slice(0, tensors[i].sizes()[1])});
        slice.copy_(tensors[i]);
    }

    DORADO_LOG(trace, "size {}x{}x{} numelem {} sum {}", tensors.size(), max_length, max_reads,
               batch.numel(), batch.sum().item<T>());

    return batch;
}

int calculate_batch_size(const std::string& device, float memory_fraction);

ModelConfig parse_model_config(const std::filesystem::path& config_path);

}  // namespace dorado::correction
