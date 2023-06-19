#pragma once

#include "read_pipeline/ReadPipeline.h"

#include <memory>
#include <vector>

class MessageSinkToVector : public dorado::MessageSink {
public:
    MessageSinkToVector(size_t max_messages, std::vector<dorado::Message>& messages) : MessageSink(max_messages), m_messages(messages) {
        m_worker_thread = std::make_unique<std::thread>(std::thread(&MessageSinkToVector::worker_thread, this));
    }
    ~MessageSinkToVector() {
        terminate();
        m_worker_thread->join();
    }

private:
    std::unique_ptr<std::thread> m_worker_thread;
    std::vector<dorado::Message>& m_messages;

    void worker_thread() {
        dorado::Message message;
        while (m_work_queue.try_pop(message)) {
            m_messages.push_back(std::move(message));
        }
    }
};

template<class T>
std::vector<T> ConvertMessages(std::vector<dorado::Message>& messages) {
    std::vector<T> converted_messages;
    for (auto& message : messages) {
        converted_messages.push_back(std::get<T>(std::move(message)));
    }
    return converted_messages;
}

/*template <typename T>
class MessageSinkToVector : public dorado::MessageSink {
public:
    MessageSinkToVector(size_t max_messages) : MessageSink(max_messages) {}

    std::vector<T> get_messages() {
        std::vector<T> vec;
        dorado::Message message;
        while (m_work_queue.try_pop(message)) {
            vec.push_back(std::get<T>(std::move(message)));
        }
        return vec;
    }

    /// Wait for @a count messages to be produced by the source.
    std::vector<T> wait_for_messages(std::size_t count) {
        std::vector<T> msgs(count);
        for (T &msg : msgs) {
            std::cerr << "Received message\n";
            dorado::Message message;
            if (!m_work_queue.try_pop(message)) {
                throw std::runtime_error("Sink was terminated early");
            }
            msg = std::get<T>(std::move(message));
        }
        return msgs;
    }
};

// Template specialization to allow messages of all types
// Useful when testing nodes that may return several different types
// TODO -- reexamine the purpose of this, or use of templates here in general.
template <>
class MessageSinkToVector<dorado::Message> : public dorado::MessageSink {
public:
    MessageSinkToVector(size_t max_messages) : MessageSink(max_messages) {}

    std::vector<dorado::Message> get_messages() {
        std::vector<dorado::Message> vec;
        dorado::Message message;
        while (m_work_queue.try_pop(message)) {
            vec.push_back(std::move(message));
        }
        return vec;
    }

    /// Wait for @a count messages to be produced by the source.
    std::vector<dorado::Message> wait_for_messages(std::size_t count) {
        std::vector<dorado::Message> msgs(count);
        for (dorado::Message &msg : msgs) {
            std::cerr << "Received message\n";
            dorado::Message message;
            if (!m_work_queue.try_pop(message)) {
                throw std::runtime_error("Sink was terminated early");
            }
            msg = std::move(message);
        }
        return msgs;
    }
};*/