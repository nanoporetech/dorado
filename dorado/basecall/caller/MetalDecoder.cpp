#include "MetalDecoder.h"

#include "decode/Decoder.h"
#include "decode/beam_search.h"
#include "torch_utils/metal_utils.h"

#include <ATen/TensorIndexing.h>
#include <ATen/ops/empty.h>

#include <cassert>

using at::indexing::Slice;
using namespace dorado::utils;

namespace {
CREATE_POINT_OF_INTEREST_ID(MetalDecoder);
}

namespace dorado::basecall {

MetalDecoder::MetalDecoder(const config::BasecallModelConfig& model_config,
                           MTL::Device* device,
                           int out_chunk_size,
                           int batch_size,
                           int out_split,
                           at::ScalarType scores_dtype,
                           at::ScalarType posts_dtype,
                           float score_scale,
                           ScanMode scan_mode)
        : m_device(device),
          m_out_chunk_size(out_chunk_size),
          m_batch_size(batch_size),
          m_out_split(out_split),
          m_out_batch_size(batch_size / out_split),
          m_states(static_cast<int>(decode::create_decoder_config(model_config).num_states())),
          m_score_scale(score_scale),
          m_options(decode::create_decoder_config(model_config).options) {
    assert(m_out_chunk_size > 0);
    assert(m_batch_size > 0);
    assert(m_out_split > 0);
    assert(m_batch_size % m_out_split == 0);

    const char* bwd_kernel = scan_mode == ScanMode::Float ? "backward_scan_float" : "backward_scan";
    const char* fwd_kernel = scan_mode == ScanMode::Float ? "forward_scan_add_softmax_float"
                                                          : "forward_scan_add_softmax";
    m_bwd_scan_cps = make_cps(m_device, bwd_kernel, {}, std::nullopt);
    m_fwd_scan_add_softmax_cps = make_cps(m_device, fwd_kernel, {}, std::nullopt);

    m_scores_TNC.reserve(m_out_split);
    m_posts_NTC.reserve(m_out_split);
    m_bwd_NTC.reserve(m_out_split);
    for (int i = 0; i < m_out_split; ++i) {
        m_scores_TNC.push_back(at::empty({m_out_chunk_size, m_out_batch_size, model_config.outsize},
                                         scores_dtype));
        m_posts_NTC.push_back(
                at::empty({m_out_batch_size, m_out_chunk_size + 1, m_states}, posts_dtype));
        m_bwd_NTC.push_back(at::empty({m_out_batch_size, m_out_chunk_size + 1, m_states}));
    }
}

bool MetalDecoder::prepare_decode(MTL::CommandBuffer* const command_buffer, int try_count) {
    POINT_OF_INTEREST_SCOPE(MetalDecoder, prepare_decode, "try_count=%i", try_count);

    std::vector<int32_t> scan_args_data{m_out_chunk_size, m_out_batch_size, m_states};
    auto scan_args = create_vec_buffer(m_device, scan_args_data);
    name_mtl_object(scan_args, "scan_kernel_args");

    for (int i = 0; i < m_out_split; ++i) {
        launch_kernel_no_wait(m_bwd_scan_cps.get(), command_buffer,
                              {scan_args.get(), mtl_for_tensor(m_scores_TNC.at(i)),
                               mtl_for_tensor(m_bwd_NTC.at(i))},
                              {}, m_out_batch_size, m_states);
        launch_kernel_no_wait(m_fwd_scan_add_softmax_cps.get(), command_buffer,
                              {scan_args.get(), mtl_for_tensor(m_scores_TNC.at(i)),
                               mtl_for_tensor(m_bwd_NTC.at(i)), mtl_for_tensor(m_posts_NTC.at(i))},
                              {}, m_out_batch_size, m_states);
    }
    return run_command_buffer("linear/scan/softmax", command_buffer, try_count);
}

DecodedData MetalDecoder::decode(int chunk_idx) const {
    POINT_OF_INTEREST_SCOPE(MetalDecoder, decode, "chunk_idx=%i", chunk_idx);

    assert(chunk_idx < m_batch_size);
    const int out_buf_idx = chunk_idx / m_out_batch_size;
    const int buf_chunk_idx = chunk_idx % m_out_batch_size;
    return decode::beam_search_decode(
            m_scores_TNC.at(out_buf_idx).index({Slice(), buf_chunk_idx}),
            m_bwd_NTC.at(out_buf_idx)[buf_chunk_idx], m_posts_NTC.at(out_buf_idx)[buf_chunk_idx],
            m_options.beam_width, m_options.beam_cut, m_options.blank_score, m_options.q_shift,
            m_options.q_scale, m_score_scale);
}

}  // namespace dorado::basecall
