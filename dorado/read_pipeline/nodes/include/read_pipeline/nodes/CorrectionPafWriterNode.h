#pragma once

#include "read_pipeline/base/MessageSink.h"

namespace dorado {

class CorrectionPafWriterNode : public MessageSink {
public:
    CorrectionPafWriterNode();
    ~CorrectionPafWriterNode();

    std::string get_name() const override;
    void terminate(const TerminateOptions&) override;
    void restart() override;

private:
    CorrectionPafWriterNode(const CorrectionPafWriterNode&) = delete;
    CorrectionPafWriterNode& operator=(const CorrectionPafWriterNode&) = delete;
    CorrectionPafWriterNode(CorrectionPafWriterNode&&) = delete;
    CorrectionPafWriterNode& operator=(CorrectionPafWriterNode&&) = delete;

private:
    void input_thread_fn();
};

}  // namespace dorado