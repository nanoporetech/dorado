#include "ModelRunner.h"

#include "crf_utils.h"
#include "decode/Decoder.h"
#include "nn/CRFModel.h"

namespace dorado::basecall {

ModelRunner::ModelRunner(const config::BasecallModelConfig &model_config, const std::string &device)
        : m_config(model_config),
          m_decoder(decode::create_decoder(device, model_config)),
          // TODO: m_options.dtype() depends on the device as TxModel uses kHalf in cuda which is not supported on CPU
          m_options(at::TensorOptions().dtype(m_decoder->dtype()).device(device)),
          m_module(load_crf_model(model_config, m_options)) {
    assert(model_config.has_normalised_basecaller_params());

    m_decoder_options.q_shift = model_config.qbias;
    m_decoder_options.q_scale = model_config.qscale;

    // Should have set batch_size to non-zero value if device == cpu
    assert(model_config.basecaller.batch_size() > 0);
    const auto N = model_config.basecaller.batch_size();
    const auto C = model_config.num_features;
    const auto T = model_config.basecaller.chunk_size();

    m_input_NCT =
            at::zeros({N, C, T}, at::TensorOptions().dtype(m_decoder->dtype()).device(at::kCPU));
}

std::vector<decode::DecodedChunk> ModelRunner::call_chunks(int num_chunks) {
    at::InferenceMode guard;
    dorado::stats::Timer timer;
    auto scores_TNC =
            m_module->forward(m_input_NCT.to(m_options.device())).transpose(0, 1).contiguous();
    const auto forward_ms = timer.GetElapsedMS();
    auto decoded_chunks = m_decoder->beam_search_part_2(
            m_decoder->beam_search_part_1({scores_TNC, num_chunks, m_decoder_options}));
    const auto forward_plus_decode_ms = timer.GetElapsedMS();
    ++m_num_batches_called;
    m_model_ms += forward_ms;
    m_decode_ms += forward_plus_decode_ms - forward_ms;
    return decoded_chunks;
}

void ModelRunner::accept_chunk(int chunk_idx, const at::Tensor &chunk_CT) {
    m_input_NCT.index_put_({chunk_idx, at::indexing::Ellipsis}, chunk_CT);
}

stats::NamedStats ModelRunner::sample_stats() const {
    stats::NamedStats stats;
    stats["batches_called"] = double(m_num_batches_called);
    stats["model_ms"] = double(m_model_ms);
    stats["decode_ms"] = double(m_decode_ms);
    return stats;
}

}  // namespace dorado::basecall
