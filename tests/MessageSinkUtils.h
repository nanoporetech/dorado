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
            vec.push_back(std::get<T>(message));
        }
        return vec;
    }
};
