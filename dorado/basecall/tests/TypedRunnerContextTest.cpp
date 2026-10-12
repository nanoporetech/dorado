#include "caller/TypedRunnerContext.h"

#include <ATen/ops/zeros.h>
#include <catch2/catch_test_macros.hpp>

#define CUT_TAG "[TypedRunnerContext]"

namespace {

class State {
public:
    State(size_t batch_size, size_t chunk_size, bool variable_chunk_sizes)
            : m_batch_size(batch_size),
              m_chunk_size(chunk_size),
              m_variable_chunk_sizes(variable_chunk_sizes) {}

    void accept_chunk(int chunk_idx, const at::Tensor& chunk) {
        m_chunk_idx = chunk_idx;
        m_chunk = chunk;
    }

    size_t batch_size() const { return m_batch_size; }
    size_t chunk_size() const { return m_chunk_size; }
    bool variable_chunk_sizes() const { return m_variable_chunk_sizes; }

    int m_chunk_idx{-1};
    at::Tensor m_chunk;

private:
    size_t m_batch_size;
    size_t m_chunk_size;
    bool m_variable_chunk_sizes;
};

class Caller {
public:
    std::vector<dorado::basecall::decode::DecodedChunk> call_chunks(State& state, int num_chunks) {
        called_state = &state;
        called_num_chunks = num_chunks;
        return std::vector<dorado::basecall::decode::DecodedChunk>(num_chunks);
    }

    State* called_state{nullptr};
    int called_num_chunks{0};
};

}  // namespace

CATCH_TEST_CASE(CUT_TAG " binds erased runner operations to typed caller state", CUT_TAG) {
    Caller caller;
    dorado::basecall::TypedRunnerContext<Caller, State> context(caller, 3, 42, true);
    dorado::basecall::ModelCaller::RunnerContext& erased_context = context;
    auto chunk = at::zeros({1, 42});

    erased_context.accept_chunk(2, chunk);
    const auto decoded_chunks = erased_context.call_chunks(2);

    CATCH_REQUIRE(caller.called_state != nullptr);
    CATCH_CHECK(caller.called_num_chunks == 2);
    CATCH_CHECK(decoded_chunks.size() == 2);
    CATCH_CHECK(caller.called_state->m_chunk_idx == 2);
    CATCH_CHECK(caller.called_state->m_chunk.is_same(chunk));
    CATCH_CHECK(erased_context.batch_size() == 3);
    CATCH_CHECK(erased_context.chunk_size() == 42);
    CATCH_CHECK(erased_context.variable_chunk_sizes());
}
