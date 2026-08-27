#pragma once

#include "read_pipeline/base/MessageSink.h"

namespace dorado {

class NullNode : public MessageSink {
public:
    // NullNode has no sink - input messages go nowhere
    NullNode();
    ~NullNode();

    std::string get_name() const override;
    void terminate(const TerminateOptions&) override;
    void restart() override;

private:
    NullNode(const NullNode&) = delete;
    NullNode& operator=(const NullNode&) = delete;
    NullNode(NullNode&&) = delete;
    NullNode& operator=(NullNode&&) = delete;

private:
    void input_thread_fn();
};

}  // namespace dorado
