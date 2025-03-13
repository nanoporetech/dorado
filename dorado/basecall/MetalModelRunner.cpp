#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "MetalModelRunner.h"

#include "MetalCaller.h"

#include <ATen/TensorIndexing.h>
#include <spdlog/spdlog.h>

namespace dorado::basecall {

MetalModelRunner::MetalModelRunner(std::shared_ptr<MetalCaller> caller)
        : m_caller(std::move(caller)), m_input(m_caller->create_input_tensor()) {}

void MetalModelRunner::accept_chunk(int chunk_idx, const at::Tensor &chunk_CT) {
    assert(config().num_features == chunk_CT.size(0));
    using at::indexing::Ellipsis;
    // Tx model input accepts NCT while LSTM models have metal convolution kernels expecting NTC
    if (config().is_lstm_model()) {
        m_input.index_put_({chunk_idx, Ellipsis, Ellipsis}, chunk_CT.transpose(0, 1));
    } else {
        m_input.index_put_({chunk_idx, Ellipsis, Ellipsis}, chunk_CT);
    }
}

std::vector<decode::DecodedChunk> MetalModelRunner::call_chunks(int num_chunks) {
    ++m_num_batches_called;
    std::vector<decode::DecodedChunk> out_chunks(num_chunks);
    m_caller->call_chunks(m_input, num_chunks, out_chunks);
    return out_chunks;
}

const config::BasecallModelConfig &MetalModelRunner::config() const { return m_caller->config(); }

size_t MetalModelRunner::chunk_size() const {
    return config().is_lstm_model() ? m_input.size(1) : m_input.size(2);
}
size_t MetalModelRunner::batch_size() const { return m_input.size(0); }

void MetalModelRunner::terminate() { m_caller->terminate(); }
void MetalModelRunner::restart() { m_caller->restart(); }

stats::NamedStats MetalModelRunner::sample_stats() const {
    stats::NamedStats stats;
    stats["batches_called"] = m_num_batches_called;
    return stats;
}

}  // namespace dorado::basecall
