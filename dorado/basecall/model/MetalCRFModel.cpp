#include "MetalCRFModel.h"

#include "torch_utils/module_utils.h"

using namespace dorado::utils;

namespace dorado::basecall::model {

MetalCRFModelImpl::MetalCRFModelImpl(const config::BasecallModelConfig &config,
                                     int chunk_size,
                                     int batch_size,
                                     int out_split,
                                     MTL::Device *const device) {
    mtl_block = register_module(
            "mtl_block", nn::metal::MetalBlock(chunk_size, batch_size, config, out_split, device));
}

void MetalCRFModelImpl::load_state_dict(const std::vector<at::Tensor> &weights) {
    utils::load_state_dict(*this, weights);
    mtl_block->load_weights();
}

MTL::CommandBuffer *MetalCRFModelImpl::forward_async(at::Tensor &in,
                                                     MTL::SharedEvent *const linear_hold_off_event,
                                                     uint64_t linear_hold_off_id,
                                                     int try_count,
                                                     std::vector<at::Tensor> &out) {
    POINT_OF_INTEREST_SCOPE(MetalCRFModel, forward_async, "try_count=%i", try_count);
    return mtl_block->forward_async(in, linear_hold_off_event, linear_hold_off_id, try_count, out);
}

}  // namespace dorado::basecall::model
