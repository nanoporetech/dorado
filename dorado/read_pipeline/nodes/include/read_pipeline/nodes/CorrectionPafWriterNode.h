#pragma once

#include "read_pipeline/base/MessageSink.h"
#include "utils/stats.h"

#include <string>

namespace dorado {

class CorrectionPafWriterNode : public MessageSink {
public:
    CorrectionPafWriterNode();
    ~CorrectionPafWriterNode();
    std::string get_name() const override { return "CorrectionPafWriterNode"; }
    void terminate(const FlushOptions &) override;
    void restart() override {
        start_input_processing([this] { input_thread_fn(); }, "paf_writer");
    }

private:
    void input_thread_fn();
};

}  // namespace dorado