#pragma once

#include "data_loader/DataLoader.h"
#include "read_pipeline/ReadPipeline.h"

#include <memory>
#include <vector>

class MessageSinkToVector : public dorado::MessageSink {
public:
    MessageSinkToVector(size_t max_messages, std::vector<dorado::Message>& messages)
            : MessageSink(max_messages), m_messages(messages) {
        m_worker_thread = std::make_unique<std::thread>(
                std::thread(&MessageSinkToVector::worker_thread, this));
    }
    ~MessageSinkToVector() { terminate_impl(); }
    void terminate() override { terminate_impl(); }

private:
    void terminate_impl() {
        terminate_input_queue();
        if (m_worker_thread->joinable()) {
            m_worker_thread->join();
        }
    }

    std::unique_ptr<std::thread> m_worker_thread;
    std::vector<dorado::Message>& m_messages;

    void worker_thread() {
        dorado::Message message;
        while (m_work_queue.try_pop(message)) {
            m_messages.push_back(std::move(message));
        }
    }
};

template <class T>
std::vector<T> ConvertMessages(std::vector<dorado::Message>& messages) {
    std::vector<T> converted_messages;
    for (auto& message : messages) {
        converted_messages.push_back(std::get<T>(std::move(message)));
    }
    return converted_messages;
}

template <class... Args>
size_t CountSinkReads(const std::string& data_path, Args&&... args) {
    dorado::PipelineDescriptor pipeline_desc;
    std::vector<dorado::Message> messages;
    pipeline_desc.add_node<MessageSinkToVector>({}, 100, messages);
    auto pipeline = dorado::Pipeline::create(std::move(pipeline_desc));

    dorado::DataLoader loader(*pipeline, args...);
    loader.load_reads(data_path, false);
    pipeline.reset();
    return messages.size();
}