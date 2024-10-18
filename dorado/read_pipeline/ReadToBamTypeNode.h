#pragma once

#include "MessageSink.h"
#include "utils/stats.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace dorado {

namespace utils {
class SampleSheet;
}

class ReadToBamTypeNode : public MessageSink {
public:
    ReadToBamTypeNode(bool emit_moves,
                      size_t num_worker_threads,
                      float modbase_threshold_frac,
                      std::unique_ptr<const utils::SampleSheet> sample_sheet,
                      size_t max_reads);
    ~ReadToBamTypeNode() { stop_input_processing(); }
    std::string get_name() const override { return "ReadToBamType"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions &) override { stop_input_processing(); };
    void restart() override {
        start_input_processing([this] { input_thread_fn(); }, "readtobam_node");
    }

private:
    void input_thread_fn();

    bool m_emit_moves;
    uint8_t m_modbase_threshold;
    std::unique_ptr<const utils::SampleSheet> m_sample_sheet;
};

}  // namespace dorado
