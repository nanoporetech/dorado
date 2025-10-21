#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <thread>

namespace dorado {

using ShutdownCallback = std::function<void()>;

class BenchmarkTimer {
public:
    BenchmarkTimer(int64_t benchmarking_period_ms, ShutdownCallback callback);

    ~BenchmarkTimer();

    void terminate();

private:
    ShutdownCallback m_shutdown_callback;
    std::atomic<bool> m_should_terminate{false};
    int64_t m_benchmarking_period_ms;
    std::chrono::time_point<std::chrono::system_clock> m_start_time;
    std::thread m_benchmarking_thread;

    void benchmarking_thread_fn();
};

}  // namespace dorado
