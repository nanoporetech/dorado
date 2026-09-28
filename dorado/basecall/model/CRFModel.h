#pragma once

#include "nn/AuxiliaryData.h"
#include "nn/CRFModules.h"
#include "nn/ConvStack.h"
#include "nn/RNNStack.h"

#include <torch/nn.h>

#include <vector>

namespace dorado::config {
struct BasecallModelConfig;
}

namespace dorado::basecall::model {

struct CRFModelImpl : torch::nn::Module {
    explicit CRFModelImpl(const config::BasecallModelConfig &config, at::TensorOptions opts);
    void load_state_dict(const std::vector<at::Tensor> &weights);
#if DORADO_CUDA_BUILD
    at::Tensor run_koi(const at::Tensor &in,
                       const nn::AuxiliaryData *aux /* = nullptr */,
                       at::Tensor *signal_embeddings /* = nullptr */);
#endif

    // When requested, signal_embeddings receives the [N, T, C] recurrent output that is input to
    // the CRF linear layers.
    at::Tensor forward(const at::Tensor &x,
                       nn::AuxiliaryData *aux /* = nullptr */,
                       at::Tensor *signal_embeddings /* = nullptr */);
    nn::ConvStack convs{nullptr};
    nn::RNNStack rnns{nullptr};
    nn::LinearLayer linear1{nullptr};
    nn::LinearCRF linear2{nullptr};
    nn::Clamp clamp1{nullptr};

protected:
    FORWARD_HAS_DEFAULT_ARGS({1, torch::nn::AnyValue(static_cast<nn::AuxiliaryData *>(nullptr))},
                             {2, torch::nn::AnyValue(static_cast<at::Tensor *>(nullptr))});
};

TORCH_MODULE(CRFModel);

}  // namespace dorado::basecall::model
