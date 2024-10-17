#pragma once

#include "MessageSink.h"

#include <string>

namespace dorado {

class NullNode : public MessageSink {
public:
    // NullNode has no sink - input messages go nowhere
    NullNode();
    ~NullNode() { stop_input_processing(); }
    std::string get_name() const override { return "NullNode"; }
    void terminate(const FlushOptions &) override { stop_input_processing(); }
    void restart() override {
        start_input_processing([this] { input_thread_fn(); }, "null_node");
    }

private:
    void input_thread_fn();
};

}  // namespace dorado
