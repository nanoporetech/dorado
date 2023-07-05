#include "ReadFilterNode.h"

#include <spdlog/spdlog.h>

namespace dorado {

void ReadFilterNode::worker_thread() {
    Message message;
    while (m_work_queue.try_pop(message)) {
        if (std::holds_alternative<CandidatePairRejectedMessage>(message)) {
            // discard, nothing downstream of this node is interested in this message
            continue;
        }

        // If this message isn't a read, we'll get a bad_variant_access exception.
        auto read = std::get<std::shared_ptr<Read>>(message);

        // Filter based on qscore.
        if ((read->calculate_mean_qscore() < m_min_qscore) ||
            read->seq.size() < m_min_read_length) {
            ++m_num_reads_filtered;
        } else if (m_read_ids_to_filter.find(read->read_id) != m_read_ids_to_filter.end()) {
            ++m_num_reads_filtered;
        } else {
            send_message_to_sink(read);
        }
    }
}

ReadFilterNode::ReadFilterNode(size_t min_qscore,
                               size_t min_read_length,
                               const std::unordered_set<std::string>& read_ids_to_filter,
                               size_t num_worker_threads)
        : MessageSink(1000),
          m_min_qscore(min_qscore),
          m_min_read_length(min_read_length),
          m_read_ids_to_filter(std::move(read_ids_to_filter)),
          m_num_reads_filtered(0) {
    for (size_t i = 0; i < num_worker_threads; i++) {
        m_workers.push_back(
                std::make_unique<std::thread>(std::thread(&ReadFilterNode::worker_thread, this)));
    }
}

void ReadFilterNode::terminate_impl() {
    terminate_input_queue();
    for (auto& m : m_workers) {
        if (m->joinable()) {
            m->join();
        }
    }
}

stats::NamedStats ReadFilterNode::sample_stats() const {
    stats::NamedStats stats = stats::from_obj(m_work_queue);
    stats["reads_filtered"] = m_num_reads_filtered;
    return stats;
}

}  // namespace dorado
