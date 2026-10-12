#include "basecall/ModelRunner.h"

#include "basecall/ModelCaller.h"
#include "basecall/crf_utils.h"
#include "decode/Decoder.h"
#include "model/CRFModel.h"

#include <sstream>

namespace dorado::basecall {

ModelRunner::ModelRunner(const config::BasecallModelConfig &model_config)
        : m_config(model_config),
          m_decoder(decode::create_decoder(c10::DeviceType::CPU, model_config)),
          m_options(at::TensorOptions().dtype(m_decoder->dtype()).device(c10::DeviceType::CPU)),
          m_module(load_crf_model(model_config, m_options)) {
    assert(model_config.has_normalised_basecaller_params());

    m_decoder_options = decode::create_decoder_options(model_config);

    // Should have set batch_size to non-zero value if device == cpu
    assert(model_config.basecaller.batch_size() > 0);
    const auto N = model_config.basecaller.batch_size();
    const auto C = model_config.num_features;
    const auto T = model_config.basecaller.chunk_size();

    m_input_NCT = at::zeros({N, C, T}, m_options);
}

ModelRunner::ModelRunner(std::shared_ptr<ModelCaller> caller, size_t batch_dims_idx)
        : m_caller(std::move(caller)),
          m_runner_context(m_caller->create_runner_context(batch_dims_idx)) {}

ModelRunner::~ModelRunner() = default;

std::vector<decode::DecodedChunk> ModelRunner::call_chunks(int num_chunks) {
    if (m_caller) {
        ++m_num_batches_called;
        return m_caller->call_chunks(*m_runner_context, num_chunks);
    }

    at::InferenceMode guard;
    dorado::stats::Timer timer;
    auto scores_TNC =
            m_module->forward(m_input_NCT.to(m_options.device())).transpose(0, 1).contiguous();
    const auto forward_ms = timer.GetElapsedMS();
    auto decoded_chunks = m_decoder->decode(
            m_decoder->prepare_decode({scores_TNC, num_chunks, m_decoder_options}));
    const auto forward_plus_decode_ms = timer.GetElapsedMS();
    ++m_num_batches_called;
    m_model_ms += forward_ms;
    m_decode_ms += forward_plus_decode_ms - forward_ms;
    return decoded_chunks;
}

void ModelRunner::accept_chunk(int chunk_idx, const at::Tensor &chunk_CT) {
    if (m_runner_context) {
        m_runner_context->accept_chunk(chunk_idx, chunk_CT);
        return;
    }
    m_input_NCT.index_put_({chunk_idx, at::indexing::Ellipsis}, chunk_CT);
}

const config::BasecallModelConfig &ModelRunner::config() const {
    return m_caller ? m_caller->config() : *m_config;
}

size_t ModelRunner::chunk_size() const {
    return m_runner_context ? m_runner_context->chunk_size() : m_input_NCT.size(2);
}

size_t ModelRunner::batch_size() const {
    return m_runner_context ? m_runner_context->batch_size() : m_input_NCT.size(0);
}

bool ModelRunner::variable_chunk_sizes() const {
    return m_runner_context && m_runner_context->variable_chunk_sizes();
}

std::pair<int, int> ModelRunner::batch_timeouts_ms() const {
    return m_caller ? m_caller->batch_timeouts_ms() : ModelRunnerBase::batch_timeouts_ms();
}

bool ModelRunner::is_low_latency() const { return m_caller && m_caller->is_low_latency(); }

void ModelRunner::terminate() {
    if (m_caller) {
        m_caller->terminate();
    }
}

void ModelRunner::restart() {
    if (m_caller) {
        m_caller->restart();
    }
}

std::string ModelRunner::get_name() const {
    std::ostringstream name;
    name << "ModelRunner_" << this;
    return name.str();
}

stats::NamedStats ModelRunner::sample_stats() const {
    stats::NamedStats stats = m_caller ? stats::from_obj(*m_caller) : stats::NamedStats{};
    stats["batches_called"] = double(m_num_batches_called);
    stats["model_ms"] = double(m_model_ms);
    stats["decode_ms"] = double(m_decode_ms);
    return stats;
}

}  // namespace dorado::basecall
