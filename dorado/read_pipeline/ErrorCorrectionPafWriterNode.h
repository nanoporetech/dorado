#pragma once
#include "read_pipeline/ReadPipeline.h"
#include "utils/stats.h"

#include <cstdint>
#include <string>

namespace dorado {

class ErrorCorrectionPafWriterNode : public MessageSink {
public:
    ErrorCorrectionPafWriterNode();
    ~ErrorCorrectionPafWriterNode();
    std::string get_name() const override { return "ErrorCorrectionPafWriterNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions &) override;
    void restart() override {
        start_input_processing([this] { input_thread_fn(); }, "paf_writer");
    }

private:
    void input_thread_fn();
};

}  // namespace dorado
