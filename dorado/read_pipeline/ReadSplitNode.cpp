#include "ReadSplitNode.h"

#include "splitter/ReadSplitter.h"

using namespace dorado::splitter;

namespace dorado {

void ReadSplitNode::input_thread_fn() {
    at::InferenceMode inference_mode_guard;

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

ReadSplitNode::ReadSplitNode(std::unique_ptr<const ReadSplitter> splitter,
                             int num_worker_threads,
                             size_t max_reads)
        : MessageSink(max_reads, num_worker_threads), m_splitter(std::move(splitter)) {
    start_input_processing(&ReadSplitNode::input_thread_fn, this);
}

stats::NamedStats ReadSplitNode::sample_stats() const { return stats::from_obj(m_work_queue); }

}  // namespace dorado
