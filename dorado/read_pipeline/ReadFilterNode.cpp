#include "ReadFilterNode.h"

#include <spdlog/spdlog.h>

namespace dorado {

void ReadFilterNode::input_thread_fn() {
    at::InferenceMode inference_mode_guard;

    Message message;
    while (get_input_message(message)) {
        // If this message isn't a read, just forward it to the sink.
        if (!is_read_message(message)) {
            send_message_to_sink(std::move(message));
            continue;
        }

        const auto& read_common = get_read_common_data(message);

        auto log_filtering = [&]() {
            if (read_common.is_duplex) {
                ++m_num_duplex_reads_filtered;
                m_num_duplex_bases_filtered += read_common.seq.length();
            } else {
                ++m_num_simplex_reads_filtered;
                m_num_simplex_bases_filtered += read_common.seq.length();
            }
        };

        // Filter based on qscore.
        if ((read_common.calculate_mean_qscore() < m_min_qscore) ||
            read_common.seq.size() < m_min_read_length ||
            (m_read_ids_to_filter.find(read_common.read_id) != m_read_ids_to_filter.end())) {
            log_filtering();
        } else {
            send_message_to_sink(std::move(message));
        }
    }
}

ReadFilterNode::ReadFilterNode(size_t min_qscore,
                               size_t min_read_length,
                               std::unordered_set<std::string> read_ids_to_filter,
                               size_t num_worker_threads)
        : MessageSink(1000, static_cast<int>(num_worker_threads)),
          m_min_qscore(min_qscore),
          m_min_read_length(min_read_length),
          m_read_ids_to_filter(std::move(read_ids_to_filter)),
          m_num_simplex_reads_filtered(0),
          m_num_duplex_reads_filtered(0) {}

stats::NamedStats ReadFilterNode::sample_stats() const {
    stats::NamedStats stats = stats::from_obj(m_work_queue);
    stats["simplex_reads_filtered"] = static_cast<double>(m_num_simplex_reads_filtered);
    stats["duplex_reads_filtered"] = static_cast<double>(m_num_duplex_reads_filtered);
    return stats;
}

}  // namespace dorado
