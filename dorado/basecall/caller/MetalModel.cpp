#include "MetalModel.h"

#include "decode/MetalDecoder.h"
#include "torch_utils/metal_utils.h"

namespace dorado::basecall {

MetalModel::MetalModel()
        : m_decode_complete_event(NS::TransferPtr(utils::get_mtl_device()->newSharedEvent())) {}

MetalModel::~MetalModel() = default;

DecodedData MetalModel::decode(int chunk_idx) const { return m_decoder->decode(chunk_idx); }

}  // namespace dorado::basecall
