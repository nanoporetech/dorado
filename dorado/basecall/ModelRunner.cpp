#include "ModelRunner.h"

#include "crf_utils.h"
#include "decode/Decoder.h"
#include "nn/CRFModel.h"

namespace dorado::basecall {

ModelRunner::ModelRunner(const CRFModelConfig &model_config,
                         const std::string &device,
                         int chunk_size,
                         int batch_size)
        : m_config(model_config),
          m_decoder(decode::create_decoder(device, model_config)),
          m_options(at::TensorOptions().dtype(m_decoder->dtype()).device(device)),
          m_module(load_crf_model(model_config, m_options)) {
    m_decoder_options.q_shift = model_config.qbias;
    m_decoder_options.q_scale = model_config.qscale;

    // adjust chunk size to be a multiple of the stride
    chunk_size -= chunk_size % model_config.stride_inner();

    m_input = at::zeros({batch_size, model_config.num_features, chunk_size},
                        at::TensorOptions().dtype(m_decoder->dtype()).device(at::kCPU));
}

std::vector<decode::DecodedChunk> ModelRunner::call_chunks(int num_chunks) {
    at::InferenceMode guard;
    dorado::stats::Timer timer;
    auto scores = m_module->forward(m_input.to(m_options.device()));
    const auto forward_ms = timer.GetElapsedMS();
    auto decoded_chunks = m_decoder->beam_search_part_2(
            m_decoder->beam_search_part_1({scores, num_chunks, m_decoder_options}));
    const auto forward_plus_decode_ms = timer.GetElapsedMS();
    ++m_num_batches_called;
    m_model_ms += forward_ms;
    m_decode_ms += forward_plus_decode_ms - forward_ms;
    return decoded_chunks;
}

void ModelRunner::accept_chunk(int chunk_idx, const at::Tensor &chunk) {
    m_input.index_put_({chunk_idx, at::indexing::Ellipsis}, chunk);
}

stats::NamedStats ModelRunner::sample_stats() const {
    stats::NamedStats stats;
    stats["batches_called"] = double(m_num_batches_called);
    stats["model_ms"] = double(m_model_ms);
    stats["decode_ms"] = double(m_decode_ms);
    return stats;
}

}  // namespace dorado::basecall
