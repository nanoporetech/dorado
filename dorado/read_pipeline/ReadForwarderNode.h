#pragma once

#include "MessageSink.h"

#include <functional>
#include <string>

namespace dorado {

// Sends on messages that are reads to the supplied callback.
class ReadForwarderNode : public MessageSink {
public:
    ReadForwarderNode(size_t max_reads,
                      int num_threads,
                      std::function<void(Message &&)> message_callback);
    ~ReadForwarderNode() { stop_input_processing(); }
    std::string get_name() const override { return "ReadForwarderNode"; }
    stats::NamedStats sample_stats() const override { return stats::from_obj(m_work_queue); }
    void terminate(const FlushOptions &) override { stop_input_processing(); }
    void restart() override {
        start_input_processing([this] { input_thread_fn(); }, "read_forward");
    }

private:
    void input_thread_fn();

    std::function<void(Message &&)> m_message_callback;
};

}  // namespace dorado
