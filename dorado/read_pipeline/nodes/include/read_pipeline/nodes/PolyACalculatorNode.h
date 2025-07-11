#pragma once

#include "read_pipeline/base/MessageSink.h"

#include <atomic>
#include <map>
#include <mutex>
#include <string>

namespace dorado {

class PolyACalculatorNode : public MessageSink {
public:
    PolyACalculatorNode(size_t num_worker_threads, size_t max_reads);
    ~PolyACalculatorNode();

    std::string get_name() const override;
    stats::NamedStats sample_stats() const override;
    void terminate(const TerminateOptions &) override;
    void restart() override;

private:
    void terminate_impl();
    void input_thread_fn();

    std::atomic<size_t> total_tail_lengths_called{0};
    std::atomic<int> num_called{0};
    std::atomic<int> num_not_called{0};

    std::mutex m_mutex;
    std::map<int, int> tail_length_counts;
};

}  // namespace dorado
