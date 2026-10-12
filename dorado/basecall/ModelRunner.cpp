#include "basecall/ModelRunner.h"

#include "basecall/ModelCaller.h"

#include <sstream>

namespace dorado::basecall {

ModelRunner::ModelRunner(std::shared_ptr<ModelCaller> caller, size_t batch_dims_idx)
        : m_caller(std::move(caller)),
          m_runner_context(m_caller->create_runner_context(batch_dims_idx)) {}

ModelRunner::~ModelRunner() = default;

std::vector<decode::DecodedChunk> ModelRunner::call_chunks(int num_chunks) {
    ++m_num_batches_called;
    return m_caller->call_chunks(*m_runner_context, num_chunks);
}

void ModelRunner::accept_chunk(int chunk_idx, const at::Tensor &chunk_CT) {
    m_runner_context->accept_chunk(chunk_idx, chunk_CT);
}

const config::BasecallModelConfig &ModelRunner::config() const { return m_caller->config(); }

size_t ModelRunner::chunk_size() const { return m_runner_context->chunk_size(); }

size_t ModelRunner::batch_size() const { return m_runner_context->batch_size(); }

bool ModelRunner::variable_chunk_sizes() const { return m_runner_context->variable_chunk_sizes(); }

std::pair<int, int> ModelRunner::batch_timeouts_ms() const { return m_caller->batch_timeouts_ms(); }

bool ModelRunner::is_low_latency() const { return m_caller->is_low_latency(); }

void ModelRunner::terminate() { m_caller->terminate(); }

void ModelRunner::restart() { m_caller->restart(); }

std::string ModelRunner::get_name() const {
    std::ostringstream name;
    name << "ModelRunner_" << this;
    return name.str();
}

stats::NamedStats ModelRunner::sample_stats() const {
    stats::NamedStats stats = stats::from_obj(*m_caller);
    stats["batches_called"] = double(m_num_batches_called);
    return stats;
}

}  // namespace dorado::basecall
