#pragma once
#include "ReadPipeline.h"
#include "splitter/ReadSplitter.h"
#include "utils/stats.h"

#include <memory>
#include <string>
#include <vector>

namespace dorado {

class ReadSplitNode : public MessageSink {
public:
    using SplitterSettings =
            std::variant<splitter::DuplexSplitSettings, splitter::RNASplitSettings>;
    ReadSplitNode(SplitterSettings settings, int num_worker_threads = 5, size_t max_reads = 1000);
    ~ReadSplitNode() { terminate_impl(); }
    std::string get_name() const override { return "ReadSplitNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions& flush_options) override { terminate_impl(); }
    void restart() override;

private:
    void start_threads();
    void terminate_impl();

    void worker_thread();  // Worker thread performs splitting asynchronously.

    const int m_num_worker_threads;
    std::vector<std::unique_ptr<std::thread>> m_worker_threads;

    std::unique_ptr<splitter::ReadSplitter> m_splitter;
};

}  // namespace dorado
