#pragma once

#include "basecall/ModelCaller.h"

#include <utility>

namespace dorado::basecall {

template <typename Caller, typename State>
class TypedRunnerContext final : public ModelCaller::RunnerContext {
public:
    template <typename... StateArgs>
    TypedRunnerContext(Caller& caller, StateArgs&&... state_args)
            : m_caller(caller), m_state(std::forward<StateArgs>(state_args)...) {}

    void accept_chunk(int chunk_idx, const at::Tensor& chunk) override {
        m_state.accept_chunk(chunk_idx, chunk);
    }

    std::vector<decode::DecodedChunk> call_chunks(int num_chunks) override {
        return m_caller.call_chunks(m_state, num_chunks);
    }

    size_t chunk_size() const override { return m_state.chunk_size(); }
    size_t batch_size() const override { return m_state.batch_size(); }
    bool variable_chunk_sizes() const override { return m_state.variable_chunk_sizes(); }

private:
    Caller& m_caller;
    State m_state;
};

}  // namespace dorado::basecall
