#pragma once

#include "nn/AuxiliaryData.h"

#include <torch/nn.h>

#include <memory>

namespace dorado::config {
struct BasecallModelConfig;
}

namespace dorado::basecall::model {

struct ModelOutput {
    at::Tensor scores;
    at::Tensor signal_embeddings;
};

class Model {
public:
    explicit Model(torch::nn::ModuleHolder<torch::nn::AnyModule> module);

    ModelOutput forward(const at::Tensor& input,
                        nn::AuxiliaryData* aux = nullptr,
                        bool collect_signal_embeddings = false);

private:
    torch::nn::ModuleHolder<torch::nn::AnyModule> m_module;
};

std::unique_ptr<Model> load_model(const config::BasecallModelConfig& model_config,
                                  const at::TensorOptions& options);

}  // namespace dorado::basecall::model
