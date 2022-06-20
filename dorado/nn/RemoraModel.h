#pragma once

#include <torch/torch.h>

#include <string>

torch::nn::ModuleHolder<torch::nn::AnyModule> load_remora_model(const std::string& path,
                                                                torch::TensorOptions options);

class RemoraCaller {
    constexpr static torch::ScalarType dtype = torch::kF32;
    torch::nn::ModuleHolder<torch::nn::AnyModule> m_module{nullptr};
    torch::TensorOptions m_options;
    torch::Tensor m_input_sigs;
    torch::Tensor m_input_seqs;

public:
    RemoraCaller(const std::string& model, std::string device);
};

class RemoraRunner {
    // one caller per model
    std::vector<std::shared_ptr<RemoraCaller>> m_callers;

public:
    RemoraRunner(const std::vector<std::string>& model_paths, std::string device);
};
