#pragma once

#include "ReadPipeline.h"
#include "utils/stats.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
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
    ReadFilterNode(size_t min_qscore, size_t min_read_length, size_t num_worker_threads);
    ~ReadFilterNode();
    std::string get_name() const override { return "ReadFilterNode"; }
    stats::NamedStats sample_stats() const override;

private:
    void worker_thread();

    // Async worker for writing.
    std::vector<std::unique_ptr<std::thread>> m_workers;

    size_t m_min_qscore;
    size_t m_min_read_length;
    std::atomic<int64_t> m_num_reads_filtered;
};

}  // namespace dorado
