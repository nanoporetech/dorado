#include "caller/BatchExecutor.h"

#include "TestUtils.h"
#include "config/BasecallModelConfig.h"

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <filesystem>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#define CUT_TAG "[BatchExecutor]"

namespace {

using namespace dorado::basecall;
using namespace std::chrono_literals;

class FakeBatch final : public InFlightBatch {};

struct Counters {
    std::atomic<int> concurrent_submits{0};
    std::atomic<int> max_concurrent_submits{0};
    std::atomic<int> submits{0};
    std::atomic<int> releases{0};
};

class FakeModel final : public CallerModel {
public:
    FakeModel(SchedulingPolicy policy, size_t units, std::shared_ptr<Counters> counters)
            : m_counters(std::move(counters)),
              m_policy(std::move(policy)),
              m_units(units),
              m_config(dorado::config::load_model_config(
                      std::filesystem::path(get_data_dir("model_configs")) /
                      "dna_r10.4.1_e8.2_400bps_hac@v4.2.0")) {}

    FakeModel(SchedulingPolicy policy, size_t units)
            : FakeModel(std::move(policy), units, std::make_shared<Counters>()) {};

    const dorado::config::BasecallModelConfig& config() const override { return m_config; }
    std::unique_ptr<RunnerState> create_runner_state(size_t) override { return nullptr; }
    std::string name() const override { return "FakeModel"; }
    SchedulingPolicy scheduling_policy() const override { return m_policy; }

    std::unique_ptr<InFlightBatch> prepare(RunnerState&, int) override {
        return std::make_unique<FakeBatch>();
    }

    void submit(InFlightBatch&) override {
        const int concurrent = ++m_counters->concurrent_submits;
        int max = m_counters->max_concurrent_submits.load();
        while (concurrent > max &&
               !m_counters->max_concurrent_submits.compare_exchange_weak(max, concurrent)) {
        }
        std::this_thread::sleep_for(2ms);
        --m_counters->concurrent_submits;
        ++m_counters->submits;
    }

    size_t decode_units(const InFlightBatch&) const override { return m_units; }

    void decode(InFlightBatch&, size_t unit, std::vector<decode::DecodedChunk>& out) override {
        out.at(unit).sequence = "unit" + std::to_string(unit);
        std::lock_guard lock(m_mutex);
        m_decode_threads.insert(std::this_thread::get_id());
    }

    void release(InFlightBatch&, std::vector<decode::DecodedChunk>&) override {
        ++m_counters->releases;
    }

    std::shared_ptr<Counters> m_counters;
    std::mutex m_mutex;
    std::set<std::thread::id> m_decode_threads;

private:
    SchedulingPolicy m_policy;
    size_t m_units;
    dorado::config::BasecallModelConfig m_config;
};

void check_decoded(const std::vector<decode::DecodedChunk>& decoded, size_t num_chunks) {
    CATCH_REQUIRE(decoded.size() == num_chunks);
    for (size_t i = 0; i < num_chunks; ++i) {
        CATCH_CHECK(decoded[i].sequence == "unit" + std::to_string(i));
    }
}

}  // namespace

CATCH_TEST_CASE(CUT_TAG " decodes on the requesting thread by default", CUT_TAG) {
    FakeModel model({}, 4);
    BatchExecutor executor(model);

    check_decoded(executor.execute(std::make_unique<FakeBatch>(), 4), 4);
    CATCH_CHECK(model.m_counters->submits == 1);
    CATCH_CHECK(model.m_counters->releases == 1);
    CATCH_REQUIRE(model.m_decode_threads.size() == 1);
    CATCH_CHECK(*model.m_decode_threads.begin() == std::this_thread::get_id());
}

CATCH_TEST_CASE(CUT_TAG " decodes units on the decode pool", CUT_TAG) {
    SchedulingPolicy policy;
    policy.decode_threads = 3;
    FakeModel model(policy, 8);
    BatchExecutor executor(model);

    check_decoded(executor.execute(std::make_unique<FakeBatch>(), 8), 8);
    CATCH_CHECK(model.m_counters->releases == 1);
    CATCH_CHECK(model.m_decode_threads.count(std::this_thread::get_id()) == 0);
}

CATCH_TEST_CASE(CUT_TAG " callers sharing a queue key submit one batch at a time", CUT_TAG) {
    SchedulingPolicy policy;
    policy.shared_queue_key = "batch_executor_test_shared";
    policy.release_point = ReleasePoint::on_submit_return;

    constexpr int num_callers = 3;
    constexpr int batches_per_caller = 5;
    auto counters = std::make_shared<Counters>();
    std::vector<std::unique_ptr<FakeModel>> models;
    std::vector<std::unique_ptr<BatchExecutor>> executors;
    for (int i = 0; i < num_callers; ++i) {
        models.push_back(std::make_unique<FakeModel>(policy, 1, counters));
        executors.push_back(std::make_unique<BatchExecutor>(*models.back()));
    }

    std::vector<std::thread> callers;
    callers.reserve(num_callers);
    std::vector<std::vector<dorado::basecall::decode::DecodedChunk>> all_decoded_chunks;
    all_decoded_chunks.reserve(num_callers * batches_per_caller);
    std::mutex decoded_chunks_mutex;
    for (int i = 0; i < num_callers; ++i) {
        callers.emplace_back([&, i] {
            for (int b = 0; b < batches_per_caller; ++b) {
                auto decoded_chunks = executors[i]->execute(std::make_unique<FakeBatch>(), 1);
                std::lock_guard lock(decoded_chunks_mutex);
                all_decoded_chunks.push_back(std::move(decoded_chunks));
            }
        });
    }
    for (auto& caller : callers) {
        caller.join();
    }

    for (const auto& decoded_chunks : all_decoded_chunks) {
        check_decoded(decoded_chunks, 1);
    }

    CATCH_CHECK(counters->submits == num_callers * batches_per_caller);
    CATCH_CHECK(counters->max_concurrent_submits == 1);
}

CATCH_TEST_CASE(CUT_TAG " restarts after terminate", CUT_TAG) {
    FakeModel model({}, 1);
    BatchExecutor executor(model);

    check_decoded(executor.execute(std::make_unique<FakeBatch>(), 1), 1);
    executor.terminate();
    executor.restart();
    check_decoded(executor.execute(std::make_unique<FakeBatch>(), 1), 1);
    CATCH_CHECK(model.m_counters->submits == 2);
}
