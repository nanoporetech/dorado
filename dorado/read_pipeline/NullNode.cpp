#include "NullNode.h"

#include <indicators/progress_bar.hpp>
#include <spdlog/spdlog.h>

namespace dorado {

void NullNode::worker_thread() {
    Message message;
    while (m_work_queue.try_pop(message)) {
        // Do nothing with the popped message.
    }
}

NullNode::NullNode() : MessageSink(1000) {
    int num_worker_threads = 4;
    for (size_t i = 0; i < num_worker_threads; i++) {
        m_workers.push_back(
                std::make_unique<std::thread>(std::thread(&NullNode::worker_thread, this)));
    }
}

NullNode::~NullNode() {
    terminate();
    for (auto& m : m_workers) {
        m->join();
    }
}

}  // namespace dorado
