#include "MessageSink.h"

#include "utils/thread_naming.h"

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

void MessageSink::start_input_processing(const std::function<void()> &input_thread_fn,
                                         const std::string &worker_name) {
    if (m_num_input_threads <= 0) {
        throw std::runtime_error("Attempting to start input processing with invalid thread count");
    }

    // Should only be called at construction time, or after stop_input_processing.
    if (!m_input_threads.empty()) {
        throw std::runtime_error("Input threads already started");
    }

    // The queue must be in started state before we attempt to pop an item,
    // otherwise the pop will fail and the thread will terminate.
    start_input_queue();
    for (int i = 0; i < m_num_input_threads; ++i) {
        m_input_threads.emplace_back([func = input_thread_fn, name = worker_name] {
            dorado::utils::set_thread_name(name.c_str());
            func();
        });
    }
}

// Mark the input queue as terminating, and stop input processing threads.
void MessageSink::stop_input_processing() {
    terminate_input_queue();
    for (auto &t : m_input_threads) {
        t.join();
    }
    m_input_threads.clear();
}

}  // namespace dorado