#pragma once

#include "ClientInfo.h"
#include "flush_options.h"
#include "messages.h"
#include "utils/AsyncQueue.h"
#include "utils/stats.h"

#include <atomic>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dorado {

// Base class for an object which consumes messages as part of the processing pipeline.
// Destructors of derived classes must call terminate() in order to shut down
// waits on the input queue before attempting to join input worker threads.
class MessageSink {
public:
    MessageSink(size_t max_messages, int num_input_threads);

    virtual ~MessageSink() = default;

    // StatsSampler will ignore nodes with an empty name.
    virtual std::string get_name() const { return std::string(""); }
    virtual stats::NamedStats sample_stats() const {
        return std::unordered_map<std::string, double>();
    }

    // Adds a message to the input queue.  This can block if the sink's queue is full.
    template <typename Msg>
    void push_message(Msg&& msg) {
        static_assert(!std::is_reference_v<Msg> && !std::is_const_v<Msg>,
                      "Pushed messages must be rvalues: the sink takes ownership");
        push_message_internal(Message(std::move(msg)));
    }

    // Waits until work is finished and shuts down worker threads.
    // No work can be done by the node after this returns until
    // restart is subsequently called.
    virtual void terminate(const FlushOptions& flush_options) = 0;

    // Starts or restarts the node following initial setup or a terminate call.
    virtual void restart() = 0;

protected:
    virtual bool forward_on_disconnected() const { return true; }

    // Terminates waits on the input queue.
    void terminate_input_queue() { m_work_queue.terminate(); }

    // Allows inputs again.
    void start_input_queue() { m_work_queue.restart(); }

    // Sends message to the designated sink.
    template <typename Msg>
    void send_message_to_sink(int sink_index, Msg&& message) {
        m_sinks.at(sink_index).get().push_message(std::forward<Msg>(message));
    }

    // Version for nodes with a single sink that is implicit.
    template <typename Msg>
    void send_message_to_sink(Msg&& message) {
        if (m_sinks.size() != 1) {
            throw std::runtime_error("Invalid m_sinks size");
        }
        send_message_to_sink(0, std::forward<Msg>(message));
    }

    // Pops the next input message, returning true on success.
    // If terminating, returns false.
    bool get_input_message(Message& message) {
        auto status = m_work_queue.try_pop(message);
        if (!m_sinks.empty() && forward_on_disconnected()) {
            while (status == utils::AsyncQueueStatus::Success && is_read_message(message) &&
                   get_read_common_data(message).client_info &&
                   get_read_common_data(message).client_info->is_disconnected()) {
                send_message_to_sink(0, std::move(message));
                status = m_work_queue.try_pop(message);
            }
        }
        return status == utils::AsyncQueueStatus::Success;
    }

    // Queue of work items for this node.
    utils::AsyncQueue<Message> m_work_queue;

    // Mark the input queue as active, and start input processing threads executing the
    // supplied functor.
    void start_input_processing(const std::function<void()>& input_thread_fn,
                                const std::string& worker_name);

    // Mark the input queue as terminating, and stop input processing threads.
    void stop_input_processing();

private:
    // The sinks to which this node can send messages.
    std::vector<std::reference_wrapper<MessageSink>> m_sinks;

    friend class Pipeline;
    void add_sink(MessageSink& sink);

    void push_message_internal(Message&& message);

    // Input processing threads.
    const int m_num_input_threads;
    std::vector<std::thread> m_input_threads;
};

}  // namespace dorado