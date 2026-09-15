#pragma once

#include "read_pipeline/base/MessageSink.h"
#include "utils/concurrency/AsyncExecutor.h"

#include <atomic>
#include <map>
#include <mutex>
#include <string>

namespace dorado {

class PolyACalculatorNode final : public MessageSink {
public:
    static inline constexpr std::size_t MAX_INPUT_QUEUE_SIZE{10000};
    static inline constexpr std::size_t MAX_PROCESSING_QUEUE_SIZE{MAX_INPUT_QUEUE_SIZE / 2};
    static inline constexpr char QUEUE_THREAD_NAME[] = "polya";

    PolyACalculatorNode(utils::concurrency::AsyncExecutor &&task_executor, size_t max_reads);
    ~PolyACalculatorNode();

    std::string get_name() const override;
    stats::NamedStats sample_stats() const override;
    void terminate(const TerminateOptions &) override;
    void restart() override;

private:
    PolyACalculatorNode(const PolyACalculatorNode &) = delete;
    PolyACalculatorNode &operator=(const PolyACalculatorNode &) = delete;
    PolyACalculatorNode(PolyACalculatorNode &&) = delete;
    PolyACalculatorNode &operator=(PolyACalculatorNode &&) = delete;

private:
    void terminate_impl(utils::AsyncQueueTerminateFast fast);
    void input_thread_fn();
    void process_read(SimplexRead &read);

    utils::concurrency::AsyncExecutor m_task_executor;

    std::atomic<size_t> total_tail_lengths_called{0};
    std::atomic<int> num_called{0};
    std::atomic<int> num_not_called{0};

    mutable std::mutex m_mutex;
    std::map<int, int> tail_length_counts;
};

}  // namespace dorado
