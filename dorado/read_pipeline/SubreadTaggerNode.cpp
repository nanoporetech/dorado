#include "SubreadTaggerNode.h"

#include <spdlog/spdlog.h>

#include <algorithm>

namespace dorado {

void SubreadTaggerNode::worker_thread() {
    at::InferenceMode inference_mode_guard;

    Message message;
    while (get_input_message(message)) {
        if (!is_read_message(message)) {
            spdlog::warn("SubreadTaggerNode received unexpected message type: {}.",
                         message.index());
            continue;
        }

        auto& read_common = get_read_common_data(message);
        const auto read_tag = read_common.read_tag;
        const auto split_count = read_common.split_count;
        if (read_common.is_duplex) {
            {
                std::lock_guard lock(m_duplex_reads_mutex);
                m_duplex_reads[read_tag].push_back(std::get<DuplexReadPtr>(std::move(message)));
                m_updated_read_tags.insert(read_tag);
            }
            m_check_duplex_cv.notify_one();
        } else {
            auto read = std::get<SimplexReadPtr>(std::move(message));
            if (split_count == 1 && read->num_duplex_candidate_pairs == 0) {
                // Unsplit, unpaired simplex read: pass directly to the next node
                send_message_to_sink(std::move(read));
                continue;
            }

            std::unique_lock subreads_lock(m_subread_groups_mutex);
            auto& subreads = m_subread_groups[read_tag];
            subreads.push_back(std::move(read));

            if (subreads.size() != split_count) {
                continue;
            }

            auto extracted_subreads = std::move(subreads);
            m_subread_groups.erase(read_tag);
            subreads_lock.unlock();

            auto num_expected_duplex =
                    std::accumulate(extracted_subreads.begin(), extracted_subreads.end(), size_t(0),
                                    [](const size_t& running_total, const SimplexReadPtr& subread) {
                                        return subread->num_duplex_candidate_pairs + running_total;
                                    });

            if (num_expected_duplex == 0) {
                // Got all subreads, no duplex to add
                for (auto& subread : extracted_subreads) {
                    send_message_to_sink(std::move(subread));
                }
            } else {
                {
                    std::lock_guard duplex_lock(m_duplex_reads_mutex);
                    m_full_subread_groups[read_tag] = {std::move(extracted_subreads),
                                                       std::vector<DuplexReadPtr>{}};
                    m_updated_read_tags.insert(read_tag);
                }
                m_check_duplex_cv.notify_one();
            }
        }
    }
}

void SubreadTaggerNode::check_duplex_thread() {
    while (!m_terminate.load()) {
        std::unique_lock lock(m_duplex_reads_mutex);
        m_check_duplex_cv.wait(lock,
                               [&] { return !m_updated_read_tags.empty() || m_terminate.load(); });

        if (m_updated_read_tags.empty()) {
            continue;
        }

        auto read_tag = *m_updated_read_tags.begin();
        m_updated_read_tags.erase(m_updated_read_tags.begin());

        auto subreads_it = m_full_subread_groups.find(read_tag);
        if (subreads_it == m_full_subread_groups.end()) {
            continue;
        }

        auto& subreads = subreads_it->second;
        auto& duplex_reads = m_duplex_reads[read_tag];
        for (auto duplex_read_iter = duplex_reads.begin();
             duplex_read_iter != duplex_reads.end();) {
            auto& duplex_read = *duplex_read_iter;
            duplex_read->read_common.subread_id = subreads.first.size() + subreads.second.size();
            subreads.second.push_back(std::move(duplex_read));
            duplex_read_iter = duplex_reads.erase(duplex_read_iter);
        }

        // check that all candidate pairs have been evaluated and that we have received a duplex read for all accepted candidate pairs
        auto num_duplex_candidates =
                std::accumulate(subreads.first.begin(), subreads.first.end(), size_t(0),
                                [](const size_t& running_total, const SimplexReadPtr& subread) {
                                    return subread->num_duplex_candidate_pairs + running_total;
                                });
        auto num_duplex = subreads.second.size();
        if (num_duplex_candidates != num_duplex) {
            continue;
        }

        // read group has received all of its expected duplex reads, push everything to the next node
        auto extracted_subreads = std::move(subreads);
        m_full_subread_groups.erase(read_tag);
        m_duplex_reads.erase(read_tag);
        lock.unlock();

        auto subread_count = extracted_subreads.first.size() + extracted_subreads.second.size();
        for (auto& subread : extracted_subreads.first) {
            subread->read_common.split_count = subread_count;
            send_message_to_sink(std::move(subread));
        }
        for (auto& subread : extracted_subreads.second) {
            subread->read_common.split_count = subread_count;
            send_message_to_sink(std::move(subread));
        }
    }
}

SubreadTaggerNode::SubreadTaggerNode(int num_worker_threads, size_t max_reads)
        : MessageSink(max_reads), m_num_worker_threads(num_worker_threads) {
    start_threads();
}

::dorado::stats::NamedStats SubreadTaggerNode::sample_stats() const {
    ::dorado::stats::NamedStats stats = ::dorado::stats::from_obj(m_work_queue);

    return stats;
}

void SubreadTaggerNode::start_threads() {
    m_terminate.store(false);
    m_duplex_thread = std::make_unique<std::thread>(&SubreadTaggerNode::check_duplex_thread, this);
    for (int i = 0; i < m_num_worker_threads; ++i) {
        auto worker_thread = std::make_unique<std::thread>(&SubreadTaggerNode::worker_thread, this);
        m_worker_threads.push_back(std::move(worker_thread));
    }
}

void SubreadTaggerNode::terminate_impl() {
    terminate_input_queue();

    // Wait for all the node's worker threads to terminate
    for (auto& t : m_worker_threads) {
        if (t->joinable()) {
            t->join();
        }
    }
    m_worker_threads.clear();

    m_terminate.store(true);
    m_check_duplex_cv.notify_one();

    if (m_duplex_thread->joinable()) {
        m_duplex_thread->join();
    }
    m_duplex_thread.reset();
}

void SubreadTaggerNode::restart() {
    restart_input_queue();
    start_threads();
}

}  // namespace dorado
