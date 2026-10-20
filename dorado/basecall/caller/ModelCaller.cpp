#include "basecall/ModelCaller.h"

#include "BatchExecutor.h"
#include "CallerModel.h"

namespace dorado::basecall {

ModelCaller::RunnerContext::RunnerContext(ModelCaller& caller, std::unique_ptr<RunnerState> state)
        : m_caller(caller), m_state(std::move(state)) {}

ModelCaller::RunnerContext::~RunnerContext() = default;

void ModelCaller::RunnerContext::accept_chunk(int chunk_idx, const at::Tensor& chunk) {
    m_state->accept_chunk(chunk_idx, chunk);
}

std::vector<decode::DecodedChunk> ModelCaller::RunnerContext::call_chunks(int num_chunks) {
    return m_caller.call_chunks(*m_state, num_chunks);
}

size_t ModelCaller::RunnerContext::chunk_size() const { return m_state->chunk_size(); }

size_t ModelCaller::RunnerContext::batch_size() const { return m_state->batch_size(); }

bool ModelCaller::RunnerContext::variable_chunk_sizes() const {
    return m_state->variable_chunk_sizes();
}

ModelCaller::ModelCaller(std::unique_ptr<CallerModel> model)
        : m_model(std::move(model)), m_executor(std::make_unique<BatchExecutor>(*m_model)) {}

ModelCaller::~ModelCaller() { m_executor->terminate(); }

std::unique_ptr<ModelCaller::RunnerContext> ModelCaller::create_runner_context(
        size_t batch_dims_idx) {
    return std::make_unique<RunnerContext>(*this, m_model->create_runner_state(batch_dims_idx));
}

std::vector<decode::DecodedChunk> ModelCaller::call_chunks(RunnerState& state, int num_chunks) {
    if (num_chunks == 0) {
        return {};
    }
    return m_executor->execute(m_model->prepare(state, num_chunks), num_chunks);
}

size_t ModelCaller::num_batch_dims() const { return m_model->num_batch_dims(); }

const config::BasecallModelConfig& ModelCaller::config() const { return m_model->config(); }

std::shared_ptr<ModBaseInfo> ModelCaller::modbase_info() const { return m_model->modbase_info(); }

std::pair<int, int> ModelCaller::batch_timeouts_ms() const { return m_model->batch_timeouts_ms(); }

bool ModelCaller::is_low_latency() const { return m_model->is_low_latency(); }

void ModelCaller::terminate() { m_executor->terminate(); }

void ModelCaller::restart() { m_executor->restart(); }

std::string ModelCaller::get_name() const { return m_model->name(); }

stats::NamedStats ModelCaller::sample_stats() const { return m_model->sample_stats(); }

}  // namespace dorado::basecall
