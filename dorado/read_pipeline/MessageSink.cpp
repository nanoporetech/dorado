#include "MessageSink.h"

#include <cassert>

namespace dorado {

MessageSink::MessageSink(size_t max_messages, int num_input_threads)
        : m_work_queue(max_messages), m_num_input_threads(num_input_threads) {}

void MessageSink::push_message_internal(Message &&message) {
#ifndef NDEBUG
    const auto status =
#endif
            m_work_queue.try_push(std::move(message));
    // try_push will fail if the sink has been told to terminate.
    // We do not expect to be pushing reads from this source if that is the case.
    assert(status == utils::AsyncQueueStatus::Success);
}

void MessageSink::add_sink(MessageSink &sink) { m_sinks.push_back(std::ref(sink)); }

}  // namespace dorado