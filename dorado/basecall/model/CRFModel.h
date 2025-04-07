#pragma once

#include "nn/CRFModules.h"

#include <torch/nn.h>

#include <vector>

namespace dorado::config {
struct BasecallModelConfig;
}

namespace dorado::basecall::model {

struct CRFModelImpl : torch::nn::Module {
    explicit CRFModelImpl(const config::BasecallModelConfig &config);
    void load_state_dict(const std::vector<at::Tensor> &weights);
#if DORADO_CUDA_BUILD
    at::Tensor run_koi(const at::Tensor &in);
#endif

    at::Tensor forward(const at::Tensor &x);
    nn::ConvStack convs{nullptr};
    nn::LSTMStack rnns{nullptr};
    nn::LinearCRF linear1{nullptr}, linear2{nullptr};
    nn::Clamp clamp1{nullptr};
    torch::nn::Sequential encoder{nullptr};
};

TORCH_MODULE(CRFModel);

}  // namespace dorado::basecall::model
