#pragma once

#include "read_pipeline/base/MessageSink.h"
#include "utils/concurrency/async_task_executor.h"

#include <atomic>
#include <map>
#include <mutex>
#include <string>

namespace dorado {

namespace utils::concurrency {
class MultiQueueThreadPool;
}  // namespace utils::concurrency

class PolyACalculatorNode : public MessageSink {
public:
    PolyACalculatorNode(std::shared_ptr<utils::concurrency::MultiQueueThreadPool> thread_pool,
                        utils::concurrency::TaskPriority pipeline_priority,
                        size_t max_reads);
    PolyACalculatorNode(size_t num_worker_threads, size_t max_reads);
    ~PolyACalculatorNode();

    std::string get_name() const override;
    stats::NamedStats sample_stats() const override;
    void terminate(const TerminateOptions &) override;
    void restart() override;

private:
    void terminate_impl(utils::AsyncQueueTerminateFast fast);
    void input_thread_fn();
    void process_read(SimplexRead &read);

    std::shared_ptr<utils::concurrency::MultiQueueThreadPool> m_thread_pool{};
    utils::concurrency::AsyncTaskExecutor m_task_executor;
    utils::concurrency::TaskPriority m_pipeline_priority{utils::concurrency::TaskPriority::normal};

    std::atomic<size_t> total_tail_lengths_called{0};
    std::atomic<int> num_called{0};
    std::atomic<int> num_not_called{0};

    mutable std::mutex m_mutex;
    std::map<int, int> tail_length_counts;
};

}  // namespace dorado
