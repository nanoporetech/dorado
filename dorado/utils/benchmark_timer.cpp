#include "utils/benchmark_timer.h"

#include "utils/thread_utils.h"

namespace {
constexpr int64_t TIMER_INCREMENT{1000};
}

namespace dorado {

BenchmarkTimer::BenchmarkTimer(int64_t benchmarking_period_ms, ShutdownCallback callback)
        : m_shutdown_callback(std::move(callback)),
          m_benchmarking_period_ms(benchmarking_period_ms),
          m_benchmarking_thread(&BenchmarkTimer::benchmarking_thread_fn, this) {}

BenchmarkTimer::~BenchmarkTimer() {
    if (m_benchmarking_thread.joinable()) {
        terminate();
    }
}

void BenchmarkTimer::terminate() {
    m_should_terminate = true;
    m_benchmarking_thread.join();
}

void BenchmarkTimer::benchmarking_thread_fn() {
    utils::set_thread_name("benchmarking_timer");
    m_start_time = std::chrono::system_clock::now();
    auto sleep_time = std::chrono::milliseconds(TIMER_INCREMENT);
    while (!m_should_terminate) {
        std::this_thread::sleep_for(sleep_time);
        const auto now = std::chrono::system_clock::now();
        auto elapsed_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(now - m_start_time).count();
        if (elapsed_ms > m_benchmarking_period_ms) {
            m_shutdown_callback();
            return;
        }
    }
}

}  // namespace dorado
