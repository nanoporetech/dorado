#pragma once

#include "ReadPipeline.h"

#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>

namespace dorado {

// Collect and calculate throughput related
// statistics for the pipeline to track dorado
// overall performance.
class StatsCounterNode : public MessageSink {
public:
    StatsCounterNode(MessageSink& sink, bool duplex);
    ~StatsCounterNode();

    void dump_stats();

private:
    void worker_thread();

    MessageSink& m_sink;

    // Async worker for writing.
    std::unique_ptr<std::thread> m_thread;

    std::atomic<int64_t> m_num_bases_processed;
    std::atomic<int64_t> m_num_samples_processed;
    std::atomic<int> m_num_reads_processed;

    std::chrono::time_point<std::chrono::system_clock> m_initialization_time;
    std::chrono::time_point<std::chrono::system_clock> m_end_time;

    bool m_duplex;
};

}  // namespace dorado
