#pragma once

#include "ReadPipeline.h"
#include "utils/stats.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace dorado {

/// Class to filter reads based on some criteria.
/// Currently only supports filtering based on
/// minimum Q-score, read length and read id.
class ReadFilterNode : public MessageSink {
public:
    ReadFilterNode(size_t min_qscore,
                   size_t min_read_length,
                   const std::unordered_set<std::string>& read_ids_to_filter,
                   size_t num_worker_threads);
    ~ReadFilterNode() { terminate_impl(); }
    std::string get_name() const override { return "ReadFilterNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate() override { terminate_impl(); }

private:
    void terminate_impl();
    void worker_thread();

    // Async worker for writing.
    std::vector<std::unique_ptr<std::thread>> m_workers;

    size_t m_min_qscore;
    size_t m_min_read_length;
    std::unordered_set<std::string> m_read_ids_to_filter;
    std::atomic<int64_t> m_num_reads_filtered;
};

}  // namespace dorado
