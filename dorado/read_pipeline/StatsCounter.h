#pragma once

#include "ReadPipeline.h"

#ifdef WIN32
#include <indicators/progress_bar.hpp>
#else
#include <indicators/block_progress_bar.hpp>
#endif

#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>

namespace dorado {

// Collect and calculate throughput related
// statistics for the pipeline to track dorado
// overall performance.
class StatsCounter {
public:
    StatsCounter(int total_reads, bool duplex);
    ~StatsCounter() = default;

    void add_basecalled_read(std::shared_ptr<Read> read);
    void add_written_read_id(const std::string& read_id);
    void add_filtered_read_id(const std::string& read_id);
    void dump_stats();

private:
    void worker_thread();

    // Async worker for writing.
    std::unique_ptr<std::thread> m_thread;

    std::atomic<int64_t> m_num_bases_processed;
    std::atomic<int64_t> m_num_samples_processed;
    std::atomic<int> m_num_reads_processed;
    std::atomic<int> m_num_reads_filtered;

    int m_num_reads_expected;

    std::chrono::time_point<std::chrono::system_clock> m_initialization_time;
    std::chrono::time_point<std::chrono::system_clock> m_end_time;

    bool m_duplex;

#ifdef WIN32
    indicators::ProgressBar m_progress_bar {
#else
    indicators::BlockProgressBar m_progress_bar{
#endif
        indicators::option::Stream{std::cerr}, indicators::option::BarWidth{30},
                indicators::option::ShowElapsedTime{true},
                indicators::option::ShowRemainingTime{true},
                indicators::option::ShowPercentage{true},
    };

    std::atomic<bool> m_terminate{false};

    std::unordered_set<std::string> m_processed_read_ids;

    std::mutex m_reads_mutex;
};

}  // namespace dorado
