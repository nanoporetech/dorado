#pragma once

#include "read_pipeline/MessageSink.h"
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
    void restart() override { start_input_processing(&ReadSplitNode::input_thread_fn, this); }

private:
    void input_thread_fn();  // Worker thread performs splitting asynchronously.

    void update_read_counters(std::size_t num_split_reads);

    std::unique_ptr<const splitter::ReadSplitter> m_splitter;

#ifdef _MSC_VER
    // Disable warning C4324: 'dorado::ReadSplitNode': structure was padded due to alignment specifier
    // as we do not care about the increased size of the ReadSplitNode.
#pragma warning(push)
#pragma warning(disable : 4324)
#endif
    alignas(std::hardware_destructive_interference_size) std::atomic_size_t
            m_num_input_reads_pushed{};
    alignas(std::hardware_destructive_interference_size) std::atomic_size_t m_num_reads_split{};
    alignas(std::hardware_destructive_interference_size) std::atomic_size_t
            m_total_num_reads_pushed{};
#ifdef _MSC_VER
#pragma warning(pop)
#endif
};

}  // namespace dorado
