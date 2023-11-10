#include "ReadFilterNode.h"

#include <spdlog/spdlog.h>

namespace dorado {

void ReadFilterNode::worker_thread() {
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
        : MessageSink(1000),
          m_num_worker_threads(num_worker_threads),
          m_min_qscore(min_qscore),
          m_min_read_length(min_read_length),
          m_read_ids_to_filter(std::move(read_ids_to_filter)),
          m_num_simplex_reads_filtered(0),
          m_num_duplex_reads_filtered(0) {
    start_threads();
}

void ReadFilterNode::start_threads() {
    for (size_t i = 0; i < m_num_worker_threads; ++i) {
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
    m_workers.clear();
}

void ReadFilterNode::restart() {
    restart_input_queue();
    start_threads();
}

stats::NamedStats ReadFilterNode::sample_stats() const {
    stats::NamedStats stats = stats::from_obj(m_work_queue);
    stats["simplex_reads_filtered"] = double(m_num_simplex_reads_filtered);
    stats["duplex_reads_filtered"] = double(m_num_duplex_reads_filtered);
    return stats;
}

}  // namespace dorado
