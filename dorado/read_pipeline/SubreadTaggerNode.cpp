#include "SubreadTaggerNode.h"

#include "utils/thread_naming.h"

#include <spdlog/spdlog.h>

#include <algorithm>

namespace dorado {

void SubreadTaggerNode::input_thread_fn() {
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
            std::lock_guard lock(m_duplex_reads_mutex);
            m_duplex_reads[read_tag].push_back(std::get<DuplexReadPtr>(std::move(message)));
            m_updated_read_tags.insert(read_tag);
        } else {
            auto read = std::get<SimplexReadPtr>(std::move(message));
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
                for (auto& subread : extracted_subreads) {
                    send_message_to_sink(std::move(subread));
                }
                continue;
            }

            std::lock_guard duplex_lock(m_duplex_reads_mutex);
            m_full_subread_groups[read_tag] = {std::move(extracted_subreads), num_expected_duplex};
            m_updated_read_tags.insert(read_tag);
        }
        // if we've got this far then we either added a duplex read or filled a group of split reads
        // so we need to check if we've received everything for that read_tag
        m_check_duplex_cv.notify_one();
    }
}

void SubreadTaggerNode::check_duplex_thread() {
    utils::set_thread_name("subread_duplex");
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

        auto& [subreads, num_expected_duplex] = subreads_it->second;
        // check that all candidate pairs have been evaluated and that we have received a duplex read for all accepted candidate pairs
        auto& duplex_reads = m_duplex_reads[read_tag];
        if (duplex_reads.size() != num_expected_duplex) {
            continue;
        }

        // received all of expected duplex reads for read group, push everything to the next node
        auto extracted_subreads = std::move(subreads);
        auto extracted_duplex_reads = std::move(duplex_reads);
        m_full_subread_groups.erase(read_tag);
        m_duplex_reads.erase(read_tag);
        lock.unlock();

        auto base = extracted_subreads.size();
        auto subread_count = base + extracted_duplex_reads.size();

        for (auto& subread : extracted_subreads) {
            subread->read_common.split_count = subread_count;
            send_message_to_sink(std::move(subread));
        }

        size_t index = 0;
        for (auto& duplex_read : extracted_duplex_reads) {
            duplex_read->read_common.split_count = subread_count;
            duplex_read->read_common.subread_id = base + index++;
            send_message_to_sink(std::move(duplex_read));
        }
    }
}

SubreadTaggerNode::SubreadTaggerNode(int num_worker_threads, size_t max_reads)
        : MessageSink(max_reads, num_worker_threads) {}

::dorado::stats::NamedStats SubreadTaggerNode::sample_stats() const {
    ::dorado::stats::NamedStats stats = ::dorado::stats::from_obj(m_work_queue);

    return stats;
}

void SubreadTaggerNode::start_threads() {
    m_terminate.store(false);
    m_duplex_thread = std::thread([this] { check_duplex_thread(); });
    start_input_processing([this] { input_thread_fn(); }, "subread_tagger");
}

void SubreadTaggerNode::terminate_impl() {
    stop_input_processing();

    m_terminate.store(true);
    m_check_duplex_cv.notify_one();

    if (m_duplex_thread.joinable()) {
        m_duplex_thread.join();
    }
}

}  // namespace dorado
