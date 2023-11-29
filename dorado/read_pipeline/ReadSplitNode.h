#pragma once
#include "ReadPipeline.h"
#include "utils/stats.h"

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
    ~ReadSplitNode() { terminate_impl(); }
    std::string get_name() const override { return "ReadSplitNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions &) override { terminate_impl(); }
    void restart() override;

private:
    void start_threads();
    void terminate_impl();

    void worker_thread();  // Worker thread performs splitting asynchronously.

    const int m_num_worker_threads;
    std::vector<std::thread> m_worker_threads;

    std::unique_ptr<const splitter::ReadSplitter> m_splitter;
};

}  // namespace dorado
