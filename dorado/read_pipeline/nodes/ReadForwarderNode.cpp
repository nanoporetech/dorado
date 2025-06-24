#include "read_pipeline/nodes/ReadForwarderNode.h"

namespace dorado {

void ReadForwarderNode::input_thread_fn() {
    Message message;
    while (get_input_message(message)) {
        if (is_read_message(message)) {
            m_message_callback(std::move(message));
        }
    }
}

ReadForwarderNode::ReadForwarderNode(size_t max_reads,
                                     int num_threads,
                                     std::function<void(Message &&)> message_callback)
        : MessageSink(max_reads, num_threads), m_message_callback(std::move(message_callback)) {}

ReadForwarderNode::~ReadForwarderNode() {
    stop_input_processing(utils::AsyncQueueTerminateFast::Yes);
}

std::string ReadForwarderNode::get_name() const { return "ReadForwarderNode"; }

void ReadForwarderNode::terminate(const TerminateOptions &terminate_options) {
    stop_input_processing(utils::terminate_fast(terminate_options.fast));
}

void ReadForwarderNode::restart() {
    start_input_processing([this] { input_thread_fn(); }, "read_forward");
}

}  // namespace dorado
