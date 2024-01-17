#pragma once

#include "read_pipeline/MessageSink.h"
#include "utils/stats.h"

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
    ~ReadSplitNode() { stop_input_processing(); }
    std::string get_name() const override { return "ReadSplitNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions &) override { stop_input_processing(); }
    void restart() override { start_input_processing(&ReadSplitNode::input_thread_fn, this); }

private:
    void input_thread_fn();  // Worker thread performs splitting asynchronously.

    std::unique_ptr<const splitter::ReadSplitter> m_splitter;
};

}  // namespace dorado
