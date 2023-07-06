#include "SubreadTaggerNode.h"

#include <algorithm>

namespace dorado {

void SubreadTaggerNode::worker_thread() {
    Message message;
    while (m_work_queue.try_pop(message)) {
        bool check_complete_groups = false;

        if (std::holds_alternative<CandidatePairRejectedMessage>(message)) {
            check_complete_groups = true;
        } else {
            // If this message isn't a read, we'll get a bad_variant_access exception.
            auto read = std::get<std::shared_ptr<Read>>(message);

            if (read->is_duplex) {
                std::unique_lock lock(m_duplex_reads_mutex);
                m_duplex_reads.push_back(std::move(read));
                lock.unlock();
                check_complete_groups = true;
            } else {
                if (read->split_count == 1 && read->num_duplex_candidate_pairs == 0) {
                    // Unsplit, unpaired simplex read: pass directly to the next node
                    send_message_to_sink(std::move(read));
                    continue;
                }

                std::lock_guard subreads_lock(m_subread_groups_mutex);
                auto& subreads = m_subread_groups[read->read_tag];
                subreads.push_back(read);

                if (subreads.size() == read->split_count) {
                    auto num_expected_duplex = std::accumulate(
                            subreads.begin(), subreads.end(), size_t(0),
                            [](const size_t& running_total, const std::shared_ptr<Read>& subread) {
                                return subread->num_duplex_candidate_pairs + running_total;
                            });

                    if (num_expected_duplex == 0) {
                        // Got all subreads, no duplex to add
                        for (auto& subread : subreads) {
                            send_message_to_sink(std::move(subread));
                        }
                    } else {
                        std::unique_lock duplex_lock(m_duplex_reads_mutex);
                        m_full_subread_groups.push_back(std::move(subreads));
                        duplex_lock.unlock();
                        check_complete_groups = true;
                    }

                    m_subread_groups.erase(read->read_tag);
                }
            }
        }

        if (check_complete_groups) {
            std::unique_lock duplex_lock(m_duplex_reads_mutex);
            for (auto subreads = m_full_subread_groups.begin();
                 subreads != m_full_subread_groups.end();) {
                for (auto duplex_read_iter = m_duplex_reads.begin();
                     duplex_read_iter != m_duplex_reads.end();) {
                    auto& duplex_read = *duplex_read_iter;
                    std::string template_read_id =
                            duplex_read->read_id.substr(0, duplex_read->read_id.find(';'));
                    uint64_t read_tag = duplex_read->read_tag;
                    // do any of the subreads match the template read id for this duplex read?
                    if (std::any_of(
                                subreads->begin(), subreads->end(),
                                [template_read_id, read_tag](const std::shared_ptr<Read>& subread) {
                                    return subread->read_id == template_read_id &&
                                           subread->read_tag == read_tag;
                                })) {
                        duplex_read->subread_id = subreads->size();
                        subreads->push_back(duplex_read);
                        duplex_read_iter = m_duplex_reads.erase(duplex_read_iter);
                    } else {
                        ++duplex_read_iter;
                    }
                }

                // check that all candidate pairs have been evaluated and that we have received a duplex read for all accepted candidate pairs
                auto num_duplex_candidates = std::accumulate(
                        subreads->begin(), subreads->end(), size_t(0),
                        [](const size_t& running_total, const std::shared_ptr<Read>& subread) {
                            return subread->num_duplex_candidate_pairs + running_total;
                        });
                auto num_duplex = std::count_if(
                        subreads->begin(), subreads->end(),
                        [](const std::shared_ptr<Read>& subread) { return subread->is_duplex; });
                if (num_duplex_candidates == num_duplex) {
                    for (auto& subread : (*subreads)) {
                        subread->split_count = subreads->size();
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
        : MessageSink(max_reads) {
    for (int i = 0; i < num_worker_threads; i++) {
        std::unique_ptr<std::thread> worker_thread =
                std::make_unique<std::thread>(&SubreadTaggerNode::worker_thread, this);
        worker_threads.push_back(std::move(worker_thread));
    }
}

void SubreadTaggerNode::terminate_impl() {
    terminate_input_queue();

    // Wait for all the node's worker threads to terminate
    for (auto& t : worker_threads) {
        if (t->joinable()) {
            t->join();
        }
    }
}

}  // namespace dorado
