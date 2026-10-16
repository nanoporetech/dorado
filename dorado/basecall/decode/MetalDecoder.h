#pragma once

#include "basecall/DecodedChunk.h"
#include "caller/MetalCaller.h"
#include "torch_utils/metal_utils.h"

#include <ATen/core/TensorBody.h>

#include <memory>
#include <vector>

namespace dorado::config {
struct BasecallModelConfig;
}

namespace dorado::basecall {

class MetalDecoder {
public:
    enum class ScanMode { Float, Quantized };

    MetalDecoder(const config::BasecallModelConfig& model_config,
                 MTL::Device* device,
                 int out_chunk_size,
                 int batch_size,
                 int out_split,
                 at::ScalarType scores_dtype,
                 at::ScalarType posts_dtype,
                 float score_scale,
                 ScanMode scan_mode);

    std::vector<at::Tensor>& scores() { return m_scores_TNC; }
    const std::vector<at::Tensor>& scores() const { return m_scores_TNC; }

    bool prepare_decode(MTL::CommandBuffer* command_buffer, int try_count);
    DecodedData decode(int chunk_idx) const;

private:
    MTL::Device* m_device;
    const int m_out_chunk_size;
    [[maybe_unused]] const int m_batch_size;
    const int m_out_split;
    const int m_out_batch_size;
    const int m_states;
    const float m_score_scale;
    decode::DecoderOptions m_options;

    std::vector<at::Tensor> m_scores_TNC;
    std::vector<at::Tensor> m_posts_NTC;
    std::vector<at::Tensor> m_bwd_NTC;
    NS::SharedPtr<MTL::ComputePipelineState> m_bwd_scan_cps;
    NS::SharedPtr<MTL::ComputePipelineState> m_fwd_scan_add_softmax_cps;
};

}  // namespace dorado::basecall
