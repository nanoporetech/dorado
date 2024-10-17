#pragma once

#include "MessageSink.h"
#include "utils/stats.h"

#include <atomic>
#include <memory>
#include <string>
#include <vector>

namespace dorado {

namespace splitter {
class ReadSplitter;
}

class ReadSplitNode : public MessageSink {
public:
    ReadSplitNode(std::unique_ptr<const splitter::ReadSplitter> splitter,
                  int num_worker_threads,
                  size_t max_reads);
    ~ReadSplitNode() { stop_input_processing(); }
    std::string get_name() const override { return "ReadSplitNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions &) override { stop_input_processing(); }
    void restart() override {
        start_input_processing([this] { input_thread_fn(); }, "readsplit_node");
    }

private:
    void input_thread_fn();  // Worker thread performs splitting asynchronously.

    void update_read_counters(std::size_t num_split_reads);

    std::unique_ptr<const splitter::ReadSplitter> m_splitter;

    std::atomic_size_t m_num_input_reads_pushed{};
    std::atomic_size_t m_num_reads_split{};
    std::atomic_size_t m_total_num_reads_pushed{};
};

}  // namespace dorado
