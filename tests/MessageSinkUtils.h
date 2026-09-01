#pragma once

#include "data_loader/DataLoader.h"
#include "read_pipeline/base/ReadPipeline.h"

#include <memory>
#include <vector>

class MessageSinkToVector : public dorado::MessageSink {
public:
    MessageSinkToVector(size_t max_messages, std::vector<dorado::Message>& messages)
            : MessageSink(max_messages, 1), m_messages(messages) {}
    ~MessageSinkToVector() { terminate_impl(dorado::utils::AsyncQueueTerminateFast::Yes); }

    std::string get_name() const override { return "sink"; }
    void terminate(const dorado::TerminateOptions& terminate_options) override {
        terminate_impl(terminate_options.fast);
    }
    void restart() override {
        start_input_processing([this] { worker_thread(); }, "MessageSinkToVector");
    }

private:
    MessageSinkToVector(const MessageSinkToVector&) = delete;
    MessageSinkToVector& operator=(const MessageSinkToVector&) = delete;
    MessageSinkToVector(MessageSinkToVector&&) = delete;
    MessageSinkToVector& operator=(MessageSinkToVector&&) = delete;

private:
    void terminate_impl(dorado::utils::AsyncQueueTerminateFast fast) {
        stop_input_processing(fast);
    }

    std::vector<dorado::Message>& m_messages;

    void worker_thread() {
        dorado::Message message;
        while (get_input_message(message)) {
            m_messages.push_back(std::move(message));
        }
    }
};

template <class T>
std::vector<T> ConvertMessages(std::vector<dorado::Message> messages) {
    std::vector<T> converted_messages;
    converted_messages.reserve(messages.size());
    for (auto& message : messages) {
        converted_messages.push_back(message.take<T>());
    }
    messages.clear();
    return converted_messages;
}

inline size_t CountSinkReads(const std::filesystem::path& data_path,
                             size_t num_worker_threads,
                             size_t max_reads,
                             std::optional<std::unordered_set<std::string>> read_list,
                             std::unordered_set<std::string> read_ignore_list) {
    dorado::PipelineDescriptor pipeline_desc;
    std::vector<dorado::Message> messages;
    pipeline_desc.add_node<MessageSinkToVector>({}, 100, messages);
    auto pipeline = dorado::Pipeline::create(std::move(pipeline_desc), nullptr);

    dorado::data_loader::DataLoader loader(*pipeline, num_worker_threads, max_reads,
                                           std::move(read_list), std::move(read_ignore_list));

    auto input_pod5_files = dorado::data_loader::InputFiles::search_pod5s(data_path, false);
    loader.load_reads(input_pod5_files, dorado::ReadOrder::UNRESTRICTED);
    pipeline->terminate({.fast = dorado::utils::AsyncQueueTerminateFast::No});
    return messages.size();
}
