#pragma once

#include "MessageSink.h"
#include "utils/stats.h"

#include <atomic>
#include <cstdint>
#include <string>
#include <unordered_set>

namespace dorado {

/// Class to filter reads based on some criteria.
/// Currently only supports filtering based on
/// minimum Q-score, read length and read id.
class ReadFilterNode : public MessageSink {
public:
    ReadFilterNode(size_t min_qscore,
                   size_t min_read_length,
                   std::unordered_set<std::string> read_ids_to_filter,
                   size_t num_worker_threads);
    ~ReadFilterNode() { stop_input_processing(); }
    std::string get_name() const override { return "ReadFilterNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions &) override { stop_input_processing(); }
    void restart() override {
        start_input_processing([this] { input_thread_fn(); }, "readfilter_node");
    }

private:
    void input_thread_fn();

    size_t m_min_qscore;
    size_t m_min_read_length;
    std::unordered_set<std::string> m_read_ids_to_filter;
    std::atomic<int64_t> m_num_simplex_reads_filtered;
    std::atomic<int64_t> m_num_simplex_bases_filtered;
    std::atomic<int64_t> m_num_duplex_reads_filtered;
    std::atomic<int64_t> m_num_duplex_bases_filtered;
};

}  // namespace dorado
