#include "read_pipeline/nodes/NullNode.h"

namespace dorado {

void NullNode::input_thread_fn() {
    Message message;
    while (get_input_message(message)) {
        // Do nothing with the popped message.
    }
}

NullNode::NullNode() : MessageSink(1000, 4) {}

NullNode::~NullNode() { stop_input_processing(utils::AsyncQueueTerminateFast::Yes); }

std::string NullNode::get_name() const { return "NullNode"; }

void NullNode::terminate(const TerminateOptions &opts) { stop_input_processing(opts.fast); }

void NullNode::restart() {
    start_input_processing([this] { input_thread_fn(); }, "null_node");
}

}  // namespace dorado
