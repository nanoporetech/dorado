#include "basecall/ModelRunner.h"

#include "TestUtils.h"
#include "basecall/ModelCaller.h"
#include "config/BasecallModelConfig.h"

#include <ATen/ops/zeros.h>
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <memory>

#define CUT_TAG "[ModelRunner]"

namespace {

class FakeCaller final : public dorado::basecall::ModelCaller {
    class Context final : public RunnerContext {
    public:
        Context(FakeCaller& caller, size_t batch_dims_idx) : m_caller(caller) {
            m_caller.created_batch_dims_idx = batch_dims_idx;
        }

        void accept_chunk(int chunk_idx, const at::Tensor& chunk) override {
            m_caller.accepted_chunk_idx = chunk_idx;
            m_caller.accepted_chunk = chunk;
        }
        size_t chunk_size() const override { return 42; }
        size_t batch_size() const override { return 3; }
        bool variable_chunk_sizes() const override { return true; }

    private:
        FakeCaller& m_caller;
    };

public:
    explicit FakeCaller(dorado::config::BasecallModelConfig config) : m_config(std::move(config)) {}

    std::unique_ptr<RunnerContext> create_runner_context(size_t batch_dims_idx) override {
        return std::make_unique<Context>(*this, batch_dims_idx);
    }

    std::vector<dorado::basecall::decode::DecodedChunk> call_chunks(RunnerContext& context,
                                                                    int num_chunks) override {
        called_context = &context;
        return std::vector<dorado::basecall::decode::DecodedChunk>(num_chunks);
    }

    size_t num_batch_dims() const override { return 1; }
    const dorado::config::BasecallModelConfig& config() const override { return m_config; }
    std::pair<int, int> batch_timeouts_ms() const override { return {11, 22}; }
    bool is_low_latency() const override { return true; }
    void terminate() override { terminated = true; }
    void restart() override { restarted = true; }
    std::string get_name() const override { return "FakeCaller"; }
    dorado::stats::NamedStats sample_stats() const override { return {{"calls", 1.}}; }

    size_t created_batch_dims_idx{0};
    int accepted_chunk_idx{-1};
    at::Tensor accepted_chunk;
    RunnerContext* called_context{nullptr};
    bool terminated{false};
    bool restarted{false};

private:
    dorado::config::BasecallModelConfig m_config;
};

dorado::config::BasecallModelConfig load_config() {
    return dorado::config::load_model_config(std::filesystem::path(get_data_dir("model_configs")) /
                                             "dna_r10.4.1_e8.2_400bps_hac@v4.2.0");
}

}  // namespace

CATCH_TEST_CASE(CUT_TAG " delegates runner operations to a shared caller context", CUT_TAG) {
    auto caller = std::make_shared<FakeCaller>(load_config());
    dorado::basecall::ModelRunner runner(caller, 7);

    CATCH_CHECK(caller->created_batch_dims_idx == 7);
    CATCH_CHECK(&runner.config() == &caller->config());
    CATCH_CHECK(runner.chunk_size() == 42);
    CATCH_CHECK(runner.batch_size() == 3);
    CATCH_CHECK(runner.variable_chunk_sizes());
    CATCH_CHECK(runner.batch_timeouts_ms() == std::pair<int, int>{11, 22});
    CATCH_CHECK(runner.is_low_latency());

    auto chunk = at::zeros({1, 42});
    runner.accept_chunk(2, chunk);
    CATCH_CHECK(caller->accepted_chunk_idx == 2);
    CATCH_CHECK(caller->accepted_chunk.is_same(chunk));

    CATCH_CHECK(runner.call_chunks(2).size() == 2);
    CATCH_CHECK(caller->called_context != nullptr);

    runner.terminate();
    runner.restart();
    CATCH_CHECK(caller->terminated);
    CATCH_CHECK(caller->restarted);
}
