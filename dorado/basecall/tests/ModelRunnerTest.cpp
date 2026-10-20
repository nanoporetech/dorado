#include "basecall/ModelRunner.h"

#include "TestUtils.h"
#include "basecall/ModelCaller.h"
#include "caller/CallerModel.h"
#include "config/BasecallModelConfig.h"

#include <ATen/ops/zeros.h>
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <filesystem>
#include <memory>

#define CUT_TAG "[ModelRunner]"

namespace {

using namespace dorado::basecall;

class FakeRunnerState final : public RunnerState {
public:
    void accept_chunk(int chunk_idx, const at::Tensor& chunk) override {
        accepted_chunk_idx = chunk_idx;
        accepted_chunk = chunk;
    }
    size_t chunk_size() const override { return 42; }
    size_t batch_size() const override { return 3; }
    bool variable_chunk_sizes() const override { return true; }

    int accepted_chunk_idx{-1};
    at::Tensor accepted_chunk;
};

class FakeBatch final : public InFlightBatch {
public:
    FakeBatch(RunnerState& state_, int num_chunks_) : state(state_), num_chunks(num_chunks_) {}
    RunnerState& state;
    int num_chunks;
};

class FakeModel final : public CallerModel {
public:
    explicit FakeModel(dorado::config::BasecallModelConfig config) : m_config(std::move(config)) {}

    const dorado::config::BasecallModelConfig& config() const override { return m_config; }
    std::unique_ptr<RunnerState> create_runner_state(size_t batch_dims_idx) override {
        created_batch_dims_idx = batch_dims_idx;
        auto state = std::make_unique<FakeRunnerState>();
        created_state = state.get();
        return state;
    }
    std::pair<int, int> batch_timeouts_ms() const override { return {11, 22}; }
    bool is_low_latency() const override { return true; }
    std::string name() const override { return "FakeModel"; }
    dorado::stats::NamedStats sample_stats() const override { return {{"calls", 1.}}; }

    std::unique_ptr<InFlightBatch> prepare(RunnerState& state, int num_chunks) override {
        prepared_state = &state;
        return std::make_unique<FakeBatch>(state, num_chunks);
    }
    void submit(InFlightBatch&) override { ++submits; }
    size_t decode_units(const InFlightBatch&) const override { return 1; }
    void decode(InFlightBatch& batch, size_t, std::vector<decode::DecodedChunk>& out) override {
        out.assign(static_cast<FakeBatch&>(batch).num_chunks,
                   decode::DecodedChunk{"fake", {}, {}, {}, {}});
    }

    size_t created_batch_dims_idx{0};
    FakeRunnerState* created_state{nullptr};
    RunnerState* prepared_state{nullptr};
    std::atomic<int> submits{0};

private:
    dorado::config::BasecallModelConfig m_config;
};

dorado::config::BasecallModelConfig load_config() {
    return dorado::config::load_model_config(std::filesystem::path(get_data_dir("model_configs")) /
                                             "dna_r10.4.1_e8.2_400bps_hac@v4.2.0");
}

}  // namespace

CATCH_TEST_CASE(CUT_TAG " delegates runner operations to the caller model", CUT_TAG) {
    auto model = std::make_unique<FakeModel>(load_config());
    auto* fake = model.get();
    auto caller = std::make_shared<ModelCaller>(std::move(model));
    ModelRunner runner(caller, 7);

    CATCH_CHECK(fake->created_batch_dims_idx == 7);
    CATCH_CHECK(&runner.config() == &fake->config());
    CATCH_CHECK(runner.chunk_size() == 42);
    CATCH_CHECK(runner.batch_size() == 3);
    CATCH_CHECK(runner.variable_chunk_sizes());
    CATCH_CHECK(runner.batch_timeouts_ms() == std::pair<int, int>{11, 22});
    CATCH_CHECK(runner.is_low_latency());

    auto chunk = at::zeros({1, 42});
    runner.accept_chunk(2, chunk);
    CATCH_CHECK(fake->created_state->accepted_chunk_idx == 2);
    CATCH_CHECK(fake->created_state->accepted_chunk.is_same(chunk));

    const auto decoded = runner.call_chunks(2);
    CATCH_REQUIRE(decoded.size() == 2);
    CATCH_CHECK(decoded[0].sequence == "fake");
    CATCH_CHECK(fake->prepared_state == fake->created_state);
    CATCH_CHECK(fake->submits == 1);
}

CATCH_TEST_CASE(CUT_TAG " can be called again after terminate and restart", CUT_TAG) {
    auto model = std::make_unique<FakeModel>(load_config());
    auto* fake = model.get();
    auto caller = std::make_shared<ModelCaller>(std::move(model));
    ModelRunner runner(caller, 0);

    CATCH_CHECK(runner.call_chunks(1).size() == 1);
    runner.terminate();
    runner.restart();
    CATCH_CHECK(runner.call_chunks(1).size() == 1);
    CATCH_CHECK(fake->submits == 2);
}

CATCH_TEST_CASE(CUT_TAG " does not run the model for an empty batch", CUT_TAG) {
    auto model = std::make_unique<FakeModel>(load_config());
    auto* fake = model.get();
    auto caller = std::make_shared<ModelCaller>(std::move(model));
    ModelRunner runner(caller, 0);

    CATCH_CHECK(runner.call_chunks(0).empty());
    CATCH_CHECK(fake->submits == 0);
}
