#pragma once

#include "MessageSink.h"
#include "utils/stats.h"

#include <string>

namespace dorado {

class CorrectionPafWriterNode : public MessageSink {
public:
    CorrectionPafWriterNode();
    ~CorrectionPafWriterNode();
    std::string get_name() const override { return "CorrectionPafWriterNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions &) override;
    void restart() override {
        start_input_processing([this] { input_thread_fn(); }, "paf_writer");
    }

private:
    void input_thread_fn();
};

}  // namespace dorado