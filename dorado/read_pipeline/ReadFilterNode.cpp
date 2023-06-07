#include "ReadFilterNode.h"

#include "utils/sequence_utils.h"

#include <spdlog/spdlog.h>

namespace dorado {

void ReadFilterNode::worker_thread() {
    m_active_threads++;

    Message message;
    while (m_work_queue.try_pop(message)) {
        // If this message isn't a read, we'll get a bad_variant_access exception.
        auto read = std::get<std::shared_ptr<Read>>(message);

        // Filter based on qscore.
        if ((utils::mean_qscore_from_qstring(read->qstring) < m_min_qscore) ||
            read->seq.size() < m_min_read_length) {
            ++m_num_reads_filtered;
            if (m_stats_counter) {
                m_stats_counter->add_filtered_read_id(read->read_id);
            }
        } else {
            m_sink.push_message(read);
        }
    }

    auto num_active_threads = --m_active_threads;
    if (num_active_threads == 0) {
        m_sink.terminate();
    }
}

ReadFilterNode::ReadFilterNode(MessageSink& sink,
                               size_t min_qscore,
                               size_t min_read_length,
                               size_t num_worker_threads,
                               StatsCounter* stats_counter)
        : MessageSink(1000),
          m_sink(sink),
          m_min_qscore(min_qscore),
          m_min_read_length(min_read_length),
          m_num_reads_filtered(0),
          m_active_threads(0),
          m_stats_counter(stats_counter) {
    for (size_t i = 0; i < num_worker_threads; i++) {
        m_workers.push_back(
                std::make_unique<std::thread>(std::thread(&ReadFilterNode::worker_thread, this)));
    }
}

ReadFilterNode::~ReadFilterNode() {
    terminate();
    for (auto& m : m_workers) {
        m->join();
    }
    m_sink.terminate();
}

stats::NamedStats ReadFilterNode::sample_stats() const {
    stats::NamedStats stats = stats::from_obj(m_work_queue);
    stats["reads_filtered"] = m_num_reads_filtered;
    return stats;
}

}  // namespace dorado
