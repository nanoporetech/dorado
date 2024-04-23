#include "ReadForwarderNode.h"

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
                                     std::function<void(Message &&)> message_callback)
        : MessageSink(max_reads, 1), m_message_callback(std::move(message_callback)) {
    start_input_processing(&ReadForwarderNode::input_thread_fn, this);
}

}  // namespace dorado
