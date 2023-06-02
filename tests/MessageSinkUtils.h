#pragma once

#include "read_pipeline/ReadPipeline.h"

template <typename T>
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
            dorado::Message message;
            if (!m_work_queue.try_pop(message)) {
                throw std::runtime_error("Sink was terminated early");
            }
            msg = std::move(message);
        }
        return msgs;
    }
};