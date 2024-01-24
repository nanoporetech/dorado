#include "NullNode.h"

namespace dorado {

void NullNode::input_thread_fn() {
    Message message;
    while (get_input_message(message)) {
        // Do nothing with the popped message.
    }
}

NullNode::NullNode() : MessageSink(1000, 4) {
    start_input_processing(&NullNode::input_thread_fn, this);
}

}  // namespace dorado
