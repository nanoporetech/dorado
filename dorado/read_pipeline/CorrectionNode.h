#pragma once

#include "read_pipeline/MessageSink.h"
#include "read_pipeline/messages.h"
#include "utils/stats.h"
#include "utils/types.h"

#include <memory>
#include <string>
#include <vector>

namespace dorado {

class CorrectionNode : public MessageSink {
public:
    CorrectionNode(int threads);
    ~CorrectionNode() { stop_input_processing(); }
    std::string get_name() const override { return "CorrectionNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions &) override { stop_input_processing(); }
    void restart() override { start_input_processing(&CorrectionNode::input_thread_fn, this); }

private:
    void input_thread_fn();
    const int m_window_size = 4096;
};

}  // namespace dorado
