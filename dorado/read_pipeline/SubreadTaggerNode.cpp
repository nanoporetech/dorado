#include "SubreadTaggerNode.h"

#include <spdlog/spdlog.h>

#include <algorithm>

namespace dorado {

void SubreadTaggerNode::worker_thread() {
    at::InferenceMode inference_mode_guard;

    Message message;
    while (get_input_message(message)) {
        bool check_complete_groups = false;

        if (is_read_message(message)) {
            auto& read_common = get_read_common_data(message);
            if (read_common.is_duplex) {
                std::unique_lock lock(m_duplex_reads_mutex);
                m_duplex_reads.push_back(std::get<DuplexReadPtr>(std::move(message)));
                lock.unlock();
                check_complete_groups = true;
            } else {
                auto read = std::get<SimplexReadPtr>(std::move(message));
                if (read_common.split_count == 1 && read->num_duplex_candidate_pairs == 0) {
                    // Unsplit, unpaired simplex read: pass directly to the next node
                    send_message_to_sink(std::move(read));
                    continue;
                }

                const auto read_tag = read_common.read_tag;
                const auto split_count = read_common.split_count;

                std::lock_guard subreads_lock(m_subread_groups_mutex);
                auto& subreads = m_subread_groups[read_tag];
                subreads.push_back(std::move(read));

                if (subreads.size() == split_count) {
                    auto num_expected_duplex = std::accumulate(
                            subreads.begin(), subreads.end(), size_t(0),
                            [](const size_t& running_total, const SimplexReadPtr& subread) {
                                return subread->num_duplex_candidate_pairs + running_total;
                            });

                    if (num_expected_duplex == 0) {
                        // Got all subreads, no duplex to add
                        for (auto& subread : subreads) {
                            send_message_to_sink(std::move(subread));
                        }
                    } else {
                        std::unique_lock duplex_lock(m_duplex_reads_mutex);
                        m_full_subread_groups.push_back(
                                {std::move(subreads), std::vector<DuplexReadPtr>{}});
                        duplex_lock.unlock();
                        check_complete_groups = true;
                    }

                    m_subread_groups.erase(read_tag);
                }
            }
        } else {
            spdlog::warn("SubreadTaggerNode received unexpected message type: {}.",
                         message.index());
            continue;
        }

        if (check_complete_groups) {
            std::unique_lock duplex_lock(m_duplex_reads_mutex);
            for (auto subreads = m_full_subread_groups.begin();
                 subreads != m_full_subread_groups.end();) {
                for (auto duplex_read_iter = m_duplex_reads.begin();
                     duplex_read_iter != m_duplex_reads.end();) {
                    auto& duplex_read = *duplex_read_iter;
                    std::string template_read_id = duplex_read->read_common.read_id.substr(
                            0, duplex_read->read_common.read_id.find(';'));
                    uint64_t read_tag = duplex_read->read_common.read_tag;
                    // do any of the subreads match the template read id for this duplex read?
                    if (std::any_of(subreads->first.begin(), subreads->first.end(),
                                    [template_read_id, read_tag](const SimplexReadPtr& subread) {
                                        return subread->read_common.read_id == template_read_id &&
                                               subread->read_common.read_tag == read_tag;
                                    })) {
                        duplex_read->read_common.subread_id =
                                subreads->first.size() + subreads->second.size();
                        subreads->second.push_back(std::move(duplex_read));
                        duplex_read_iter = m_duplex_reads.erase(duplex_read_iter);
                    } else {
                        ++duplex_read_iter;
                    }
                }

                // check that all candidate pairs have been evaluated and that we have received a duplex read for all accepted candidate pairs
                auto num_duplex_candidates = std::accumulate(
                        subreads->first.begin(), subreads->first.end(), size_t(0),
                        [](const size_t& running_total, const SimplexReadPtr& subread) {
                            return subread->num_duplex_candidate_pairs + running_total;
                        });
                auto num_duplex = subreads->second.size();
                if (num_duplex_candidates == num_duplex) {
                    auto subread_count = subreads->first.size() + subreads->second.size();
                    for (auto& subread : subreads->first) {
                        subread->read_common.split_count = subread_count;
                        send_message_to_sink(std::move(subread));
                    }
                    for (auto& subread : subreads->second) {
                        subread->read_common.split_count = subread_count;
                        send_message_to_sink(std::move(subread));
                    }
                    subreads = m_full_subread_groups.erase(subreads);
                } else {
                    ++subreads;
                }
            }
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
}

void SubreadTaggerNode::restart() {
    restart_input_queue();
    start_threads();
}

}  // namespace dorado
