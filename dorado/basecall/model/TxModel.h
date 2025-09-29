#pragma once

#include "nn/AuxiliaryData.h"
#include "nn/ConvStack.h"
#include "nn/LinearUpsample.h"
#include "nn/TxModules.h"
#include "torch_utils/module_utils.h"

#include <c10/core/Device.h>
#include <c10/core/TensorOptions.h>
#include <torch/nn.h>

#include <memory>
#include <vector>

namespace dorado::config {
struct BasecallModelConfig;
}

namespace dorado::basecall::model {

struct TxModelImpl : torch::nn::Module {
    explicit TxModelImpl(const config::BasecallModelConfig &config,
                         const at::TensorOptions &options);

    void load_state_dict(const std::vector<at::Tensor> &weights) {
        utils::load_state_dict(*this, weights);
    }

    at::Tensor forward(const at::Tensor &chunk_NCT, nn::AuxiliaryData *aux /* = nullptr */);

    nn::ConvStack convs{nullptr};
    nn::TxEncoderStack tx_encoder{nullptr};
    nn::LinearUpsample tx_decoder{nullptr};
    nn::LinearScaledCRF crf{nullptr};

    const at::TensorOptions m_options;

protected:
    FORWARD_HAS_DEFAULT_ARGS({1, torch::nn::AnyValue(static_cast<nn::AuxiliaryData *>(nullptr))});
};

TORCH_MODULE(TxModel);

}  // namespace dorado::basecall::model
