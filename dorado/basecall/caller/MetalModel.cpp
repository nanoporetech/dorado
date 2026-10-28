#include "MetalModel.h"

#include "decode/MetalDecoder.h"
#include "torch_utils/metal_utils.h"
#include "utils/log_utils.h"
#include "utils/memory_utils.h"

namespace dorado::basecall {

MetalModel::MetalModel(const config::BasecallModelConfig& model_config) : m_config(model_config) {
    utils::ScopedAutoReleasePool autorelease_pool;

    m_device = utils::get_mtl_device();
    m_decode_complete_event = NS::TransferPtr(m_device->newSharedEvent());

    // Our metal builds assume shared memory, so it's safe to check host.
    if (auto total_mem = utils::total_host_memory_GB(); total_mem < 16) {
        DORADO_LOG(warn,
                   "Less than 16GB of memory available: {}GB detected. "
                   "This is below minimum spec and may cause issues",
                   total_mem);
    }
}

MetalModel::~MetalModel() = default;

DecodedData MetalModel::decode(int chunk_idx) const { return m_decoder->decode(chunk_idx); }

}  // namespace dorado::basecall
