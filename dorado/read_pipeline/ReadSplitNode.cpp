#include "ReadSplitNode.h"

#include "splitter/ReadSplitter.h"

#include <spdlog/spdlog.h>

#include <algorithm>

using namespace dorado::splitter;

namespace dorado {

namespace {

void validate_split_results(const std::vector<SimplexReadPtr>& split_reads,
                            const std::string& original_read_id,
                            uint64_t tag) {
    // INSTX-5275 The server relies on a each input read (as identified by read_tag)
    // being returned, and the number of reads with that read_tag should match the
    // split_count otherwise the read will never be returned to the client
    // causing clients hangs at 99% as they wait for the missing read(s).
    std::size_t matches =
            std::count_if(split_reads.begin(), split_reads.end(),
                          [tag](const auto& read) { return read->read_common.read_tag == tag; });
    if (matches == 0 || matches < split_reads.size() ||
        matches != split_reads[0]->read_common.split_count) {
        spdlog::error("ReadSplitNode failed to forward read id: {}", original_read_id);
        throw std::runtime_error("ReadSplitNode failed to forward read id: " + original_read_id);
    }
}

}  // namespace

void ReadSplitNode::update_read_counters(std::size_t num_split_reads) {
    m_total_num_reads_pushed.fetch_add(num_split_reads, std::memory_order_relaxed);
    m_num_input_reads_pushed.fetch_add(1, std::memory_order_relaxed);
    if (num_split_reads > 1) {
        m_num_reads_split.fetch_add(1, std::memory_order_relaxed);
    }
}

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
        auto initial_read = std::get<SimplexReadPtr>(std::move(message));
        auto read_id = initial_read->read_common.read_id;
        auto tag = initial_read->read_common.read_tag;

        auto split_reads = m_splitter->split(std::move(initial_read));
        validate_split_results(split_reads, read_id, tag);

        for (auto& subread : split_reads) {
            //TODO correctly process end_reason when we have them
            send_message_to_sink(std::move(subread));
        }

        update_read_counters(split_reads.size());
    }
}

ReadSplitNode::ReadSplitNode(std::unique_ptr<const ReadSplitter> splitter,
                             int num_worker_threads,
                             size_t max_reads)
        : MessageSink(max_reads, num_worker_threads), m_splitter(std::move(splitter)) {}

stats::NamedStats ReadSplitNode::sample_stats() const {
    auto stats = stats::from_obj(m_work_queue);
    stats["num_input_reads_pushed"] = static_cast<double>(m_num_input_reads_pushed.load());
    stats["num_reads_split"] = static_cast<double>(m_num_reads_split.load());
    stats["total_num_reads_pushed"] = static_cast<double>(m_total_num_reads_pushed.load());
    return stats;
}

}  // namespace dorado
