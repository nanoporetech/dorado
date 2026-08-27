#pragma once

#include "read_pipeline/base/MessageSink.h"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>

namespace dorado::batchsize_benchmarks {

class BenchmarkSink final : public MessageSink {
    using Clock = std::chrono::steady_clock;

public:
    BenchmarkSink();
    ~BenchmarkSink();

    std::string get_name() const override;
    void terminate(const TerminateOptions &terminate_options) override;
    void restart() override;

    // Wait until the first batch of reads make it to the sink.
    // Returns how long we had to wait for the first batch to appear.
    Clock::duration wait_for_reads();

    // Determine how long the pipeline has been running since last restart.
    Clock::duration elapsed();

private:
    BenchmarkSink(const BenchmarkSink &) = delete;
    BenchmarkSink &operator=(const BenchmarkSink &) = delete;
    BenchmarkSink(BenchmarkSink &&) = delete;
    BenchmarkSink &operator=(BenchmarkSink &&) = delete;

private:
    void input_thread_fn();

private:
    std::mutex m_first_read_mutex;
    std::condition_variable m_first_read_cv;
    std::optional<Clock::duration> m_first_read_batch_time;

    Clock::time_point m_pipeline_start_time;
};

}  // namespace dorado::batchsize_benchmarks
