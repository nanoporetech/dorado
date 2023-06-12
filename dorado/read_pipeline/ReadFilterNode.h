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
/// Currently only supports one baked in type of
/// filtering based on qscore.
/// TODO: Should be replaced with more general
/// mechanism to define and pass in arbitrary filters
/// which are applied to each read.
class ReadFilterNode : public MessageSink {
public:
    ReadFilterNode(MessageSink& sink,
                   size_t min_qscore,
                   size_t min_read_length,
                   const std::unordered_set<std::string>& read_ids_to_filter,
                   size_t num_worker_threads);
    ~ReadFilterNode();
    std::string get_name() const override { return "ReadFilterNode"; }
    stats::NamedStats sample_stats() const override;

private:
    MessageSink& m_sink;
    void worker_thread();

    // Async worker for writing.
    std::vector<std::unique_ptr<std::thread>> m_workers;
    std::atomic<size_t> m_active_threads;

    size_t m_min_qscore;
    size_t m_min_read_length;
    std::unordered_set<std::string> m_read_ids_to_filter;
    std::atomic<int64_t> m_num_reads_filtered;
};

}  // namespace dorado
