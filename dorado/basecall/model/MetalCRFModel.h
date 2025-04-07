#pragma once

#include "config/BasecallModelConfig.h"
#include "nn/metal/MetalModules.h"
#include "torch_utils/metal_utils.h"

#include <torch/nn.h>

#include <vector>

namespace dorado::basecall::model {

struct MetalCRFModelImpl : torch::nn::Module {
    MetalCRFModelImpl(const config::BasecallModelConfig &config,
                      int chunk_size,
                      int batch_size,
                      int out_split,
                      MTL::Device *const device);

    void load_state_dict(const std::vector<at::Tensor> &weights);

    MTL::CommandBuffer *forward_async(at::Tensor &in,
                                      MTL::SharedEvent *const linear_hold_off_event,
                                      uint64_t linear_hold_off_id,
                                      int try_count,
                                      std::vector<at::Tensor> &out);
    nn::metal::MetalBlock mtl_block{nullptr};
};

TORCH_MODULE(MetalCRFModel);

}  // namespace dorado::basecall::model
