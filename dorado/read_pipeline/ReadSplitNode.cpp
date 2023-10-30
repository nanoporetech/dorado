#include "ReadSplitNode.h"

#include "splitter/DuplexReadSplitter.h"
#include "splitter/RNAReadSplitter.h"
#include "splitter/ReadSplitter.h"

using namespace dorado::splitter;

namespace dorado {

void ReadSplitNode::worker_thread() {
    torch::InferenceMode inference_mode_guard;

    Message message;
    while (get_input_message(message)) {
        // If this message isn't a read, just forward it to the sink.
        if (!std::holds_alternative<SimplexReadPtr>(message)) {
            send_message_to_sink(std::move(message));
            continue;
        }

        // If this message isn't a read, we'll get a bad_variant_access exception.
        auto init_read = std::get<SimplexReadPtr>(std::move(message));
        for (auto& subread : m_splitter->split(std::move(init_read))) {
            //TODO correctly process end_reason when we have them
            send_message_to_sink(std::move(subread));
        }
    }
}

ReadSplitNode::ReadSplitNode(SplitterSettings settings, int num_worker_threads, size_t max_reads)
        : MessageSink(max_reads), m_num_worker_threads(num_worker_threads) {
    if (std::holds_alternative<RNASplitSettings>(settings)) {
        m_splitter =
                std::make_unique<RNAReadSplitter>(std::move(std::get<RNASplitSettings>(settings)));
    } else if (std::holds_alternative<DuplexSplitSettings>(settings)) {
        m_splitter = std::make_unique<DuplexReadSplitter>(
                std::move(std::get<DuplexSplitSettings>(settings)));
    }

    start_threads();
}

void ReadSplitNode::start_threads() {
    for (int i = 0; i < m_num_worker_threads; ++i) {
        m_worker_threads.push_back(
                std::make_unique<std::thread>(&ReadSplitNode::worker_thread, this));
    }
}

void ReadSplitNode::terminate_impl() {
    terminate_input_queue();

    // Wait for all the Node's worker threads to terminate
    for (auto& t : m_worker_threads) {
        if (t->joinable()) {
            t->join();
        }
    }
    m_worker_threads.clear();
}

void ReadSplitNode::restart() {
    restart_input_queue();
    start_threads();
}

stats::NamedStats ReadSplitNode::sample_stats() const { return stats::from_obj(m_work_queue); }

}  // namespace dorado
