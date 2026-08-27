#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <thread>

namespace dorado {

using ShutdownCallback = std::function<void()>;

class BenchmarkTimer {
public:
    explicit BenchmarkTimer(std::chrono::seconds benchmarking_period, ShutdownCallback callback);
    ~BenchmarkTimer();

    void terminate();

private:
    BenchmarkTimer(const BenchmarkTimer&) = delete;
    BenchmarkTimer& operator=(const BenchmarkTimer&) = delete;
    BenchmarkTimer(BenchmarkTimer&&) = delete;
    BenchmarkTimer& operator=(BenchmarkTimer&&) = delete;

private:
    ShutdownCallback m_shutdown_callback;
    std::atomic<bool> m_should_terminate{false};
    std::thread m_benchmarking_thread;

    void benchmarking_thread_fn(std::chrono::seconds benchmarking_period);
};

}  // namespace dorado
