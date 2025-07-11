#pragma once

#include "read_pipeline/base/MessageSink.h"

#include <atomic>
#include <memory>
#include <string>

namespace dorado {

namespace splitter {
class ReadSplitter;
}

class ReadSplitNode : public MessageSink {
public:
    ReadSplitNode(std::unique_ptr<const splitter::ReadSplitter> splitter,
                  int num_worker_threads,
                  size_t max_reads);
    ~ReadSplitNode();

    std::string get_name() const override;
    stats::NamedStats sample_stats() const override;
    void terminate(const TerminateOptions &) override;
    void restart() override;

private:
    void input_thread_fn();  // Worker thread performs splitting asynchronously.

    void update_read_counters(std::size_t num_split_reads);

    std::unique_ptr<const splitter::ReadSplitter> m_splitter;

    std::atomic_size_t m_num_input_reads_pushed{};
    std::atomic_size_t m_num_reads_split{};
    std::atomic_size_t m_total_num_reads_pushed{};
};

}  // namespace dorado
