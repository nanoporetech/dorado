#include "BatchedModelCaller.h"

namespace dorado::basecall {

namespace {

class BatchedRunnerContext final : public ModelCaller::RunnerContext {
public:
    BatchedRunnerContext(BatchedModelCaller& caller, std::unique_ptr<RunnerState> state)
            : m_caller(caller), m_state(std::move(state)) {}

    void accept_chunk(int chunk_idx, const at::Tensor& chunk) override {
        m_state->accept_chunk(chunk_idx, chunk);
    }

    std::vector<decode::DecodedChunk> call_chunks(int num_chunks) override {
        return m_caller.call_chunks(*m_state, num_chunks);
    }

    size_t chunk_size() const override { return m_state->chunk_size(); }
    size_t batch_size() const override { return m_state->batch_size(); }
    bool variable_chunk_sizes() const override { return m_state->variable_chunk_sizes(); }

private:
    BatchedModelCaller& m_caller;
    std::unique_ptr<RunnerState> m_state;
};

}  // namespace

BatchedModelCaller::BatchedModelCaller(std::unique_ptr<CallerModel> model)
        : m_model(std::move(model)), m_executor(*m_model) {}

BatchedModelCaller::~BatchedModelCaller() { m_executor.terminate(); }

std::unique_ptr<ModelCaller::RunnerContext> BatchedModelCaller::create_runner_context(
        size_t batch_dims_idx) {
    return std::make_unique<BatchedRunnerContext>(*this,
                                                  m_model->create_runner_state(batch_dims_idx));
}

std::vector<decode::DecodedChunk> BatchedModelCaller::call_chunks(RunnerState& state,
                                                                  int num_chunks) {
    if (num_chunks == 0) {
        return {};
    }
    return m_executor.execute(m_model->prepare(state, num_chunks), num_chunks);
}

}  // namespace dorado::basecall
