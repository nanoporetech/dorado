#pragma once

#include "types.h"

#include <torch/torch.h>

namespace dorado::correction {

template <typename T>
torch::Tensor collate(std::vector<torch::Tensor>& tensors,
                      T fill_val,
                      torch::ScalarType type,
                      T* mem_ptr) {
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
    auto options = torch::TensorOptions().dtype(type).device(torch::kCPU);
    //torch::Tensor batch = torch::empty({(int)tensors.size(), max_length, max_reads}, options);
    torch::Tensor batch =
            torch::from_blob(mem_ptr, {(int)tensors.size(), max_length, max_reads}, options);
    T* ptr = batch.data_ptr<T>();
    std::fill(ptr, ptr + batch.numel(), fill_val);
    // Copy over data for each tensor
    for (size_t i = 0; i < tensors.size(); i++) {
        //spdlog::info("slice {}x{}, {}x{}", 0, tensors[i].sizes()[0], 0, tensors[i].sizes()[1]);
        torch::Tensor slice = batch.index({(int)i, torch::indexing::Slice(0, tensors[i].sizes()[0]),
                                           torch::indexing::Slice(0, tensors[i].sizes()[1])});
        slice.copy_(tensors[i]);
    }
    spdlog::debug("size {}x{}x{} numelem {} sum {}", tensors.size(), max_length, max_reads,
                  batch.numel(), batch.sum().item<T>());
    return batch;
}

[[maybe_unused]] void print_size(const torch::Tensor& t, const std::string& name) {
    std::string size = "";
    for (auto s : t.sizes()) {
        size += std::to_string(s) + ",";
    }
    std::stringstream ss;
    ss << t.dtype();
    spdlog::info("{} tensor size {} dtype {}", name, size, ss.str());
}

}  // namespace dorado::correction
