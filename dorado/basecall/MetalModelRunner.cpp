#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "MetalModelRunner.h"

#include "CRFModelConfig.h"
#include "MetalCaller.h"

using namespace dorado::utils;
using torch::indexing::Ellipsis;

namespace dorado::basecall {

MetalModelRunner::MetalModelRunner(std::shared_ptr<MetalCaller> caller) : m_caller(caller) {
    // Metal convolution kernels operate with channel ordering (N, T, C).  If m_input
    // is to be submitted directly then it must also have this arrangement.
    // Note that this is not the same as other caller implementations, which
    // have T innermost.
    m_input = torch::empty(
            {caller->m_batch_size, caller->m_in_chunk_size, caller->m_num_input_features},
            torch::kF16);
}

void MetalModelRunner::accept_chunk(int chunk_idx, const at::Tensor &chunk) {
    if (chunk.dim() == 1) {
        // Input has single feature dimension.
        assert(m_caller->m_num_input_features == 1);
        m_input.index_put_({chunk_idx, Ellipsis, 0}, chunk);
    } else {
        // Chunks are passed with timestep the innermost dimension, whereas we need
        // channels innermost.
        assert(m_caller->m_num_input_features == chunk.size(0));
        m_input.index_put_({chunk_idx, Ellipsis, Ellipsis}, chunk.transpose(0, 1));
    }
}

std::vector<decode::DecodedChunk> MetalModelRunner::call_chunks(int num_chunks) {
    ++m_num_batches_called;
    std::vector<decode::DecodedChunk> out_chunks(num_chunks);
    m_caller->call_chunks(m_input, num_chunks, out_chunks);
    return out_chunks;
}

const CRFModelConfig &MetalModelRunner::config() const { return m_caller->m_config; }
size_t MetalModelRunner::model_stride() const { return m_caller->m_config.stride; }
size_t MetalModelRunner::chunk_size() const { return m_input.size(1); }
size_t MetalModelRunner::batch_size() const { return m_input.size(0); }

void MetalModelRunner::terminate() { m_caller->terminate(); }
void MetalModelRunner::restart() { m_caller->restart(); }

stats::NamedStats MetalModelRunner::sample_stats() const {
    stats::NamedStats stats;
    stats["batches_called"] = m_num_batches_called;
    return stats;
}

}  // namespace dorado::basecall
