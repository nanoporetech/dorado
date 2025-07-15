#pragma once

#include "read_pipeline/base/MessageSink.h"

#include <functional>
#include <string>

namespace dorado {

// Sends on messages that are reads to the supplied callback.
class ReadForwarderNode : public MessageSink {
public:
    ReadForwarderNode(size_t max_reads,
                      int num_threads,
                      std::function<void(Message &&)> message_callback);
    ~ReadForwarderNode();

    std::string get_name() const override;
    void terminate(const TerminateOptions &) override;
    void restart() override;

private:
    void input_thread_fn();

    std::function<void(Message &&)> m_message_callback;
};

}  // namespace dorado
