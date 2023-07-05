#include "PairingNode.h"
namespace {
bool is_within_time_and_length_criteria(const std::shared_ptr<dorado::Read>& read1,
                                        const std::shared_ptr<dorado::Read>& read2) {
    int max_time_delta_ms = 5000;
    float min_seq_len_ratio = 0.95f;
    int delta = read2->start_time_ms - read1->get_end_time_ms();
    int seq_len1 = read1->seq.length();
    int seq_len2 = read2->seq.length();
    float len_ratio = static_cast<float>(std::min(seq_len1, seq_len2)) /
                      static_cast<float>(std::max(seq_len1, seq_len2));
    return (delta >= 0) && (delta < max_time_delta_ms) && (len_ratio >= min_seq_len_ratio);
}
}  // namespace

namespace dorado {

void PairingNode::pair_list_worker_thread() {
    Message message;
    while (m_work_queue.try_pop(message)) {
        // If this message isn't a read, we'll get a bad_variant_access exception.
        auto read = std::get<std::shared_ptr<Read>>(message);

        bool read_is_template = false;
        bool partner_found = false;
        std::string partner_id;

        // Check if read is a template with corresponding complement
        std::unique_lock<std::mutex> tc_lock(m_tc_map_mutex);

        auto it = m_template_complement_map.find(read->read_id);
        if (it != m_template_complement_map.end()) {
            partner_id = it->second;
            tc_lock.unlock();
            read_is_template = true;
            partner_found = true;
        } else {
            {
                tc_lock.unlock();
                std::lock_guard<std::mutex> ct_lock(m_ct_map_mutex);
                auto it = m_complement_template_map.find(read->read_id);
                if (it != m_complement_template_map.end()) {
                    partner_id = it->second;
                    partner_found = true;
                }
            }
        }

        if (partner_found) {
            std::unique_lock<std::mutex> read_cache_lock(m_read_cache_mutex);
            auto partner_read_itr = m_read_cache.find(partner_id);
            if (partner_read_itr == m_read_cache.end()) {
                // Partner is not in the read cache
                m_read_cache.insert({read->read_id, read});
                read_cache_lock.unlock();
            } else {
                auto partner_read = partner_read_itr->second;
                m_read_cache.erase(partner_read_itr);
                read_cache_lock.unlock();

                std::shared_ptr<Read> template_read;
                std::shared_ptr<Read> complement_read;

                if (read_is_template) {
                    template_read = read;
                    complement_read = partner_read;
                } else {
                    complement_read = read;
                    template_read = partner_read;
                }

                ReadPair read_pair;
                read_pair.read_1 = template_read;
                read_pair.read_2 = complement_read;

                ++template_read->num_duplex_candidate_pairs;

                send_message_to_sink(std::make_shared<ReadPair>(read_pair));
            }
        }
    }
    --m_num_worker_threads;
}

void PairingNode::pair_generating_worker_thread() {
    auto compare_reads_by_time = [](const std::shared_ptr<Read>& read1,
                                    const std::shared_ptr<Read>& read2) {
        return read1->start_time_ms < read2->start_time_ms;
    };

    Message message;
    while (m_work_queue.try_pop(message)) {
        // If this message isn't a read, we'll get a bad_variant_access exception.
        auto read = std::get<std::shared_ptr<Read>>(message);

        int channel = read->attributes.channel_number;
        int mux = read->attributes.mux;
        std::string run_id = read->run_id;
        std::string flowcell_id = read->flowcell_id;
        int32_t client_id = read->client_id;

        std::unique_lock<std::mutex> lock(m_pairing_mtx);
        UniquePoreIdentifierKey key = std::make_tuple(channel, mux, run_id, flowcell_id, client_id);
        auto read_list_iter = m_channel_mux_read_map.find(key);
        // Check if the key is already in the list
        if (read_list_iter == m_channel_mux_read_map.end()) {
            // Key is not in the dequeue
            // Add the new key to the end of the list
            m_working_channel_mux_keys.push_back(key);
            m_channel_mux_read_map.insert({key, {read}});

            if (m_working_channel_mux_keys.size() > m_max_num_keys) {
                // Remove the oldest key (front of the list)
                auto oldest_key = m_working_channel_mux_keys.front();
                m_working_channel_mux_keys.pop_front();

                auto oldest_key_it = m_channel_mux_read_map.find(oldest_key);

                // Remove the oldest key from the map
                for (auto read_ptr : oldest_key_it->second) {
                    send_message_to_sink(read_ptr);
                }
                m_channel_mux_read_map.erase(oldest_key);
                assert(m_channel_mux_read_map.size() == m_working_channel_mux_keys.size());
            }
        } else {
            auto& cached_read_list = read_list_iter->second;
            auto later_read = std::lower_bound(cached_read_list.begin(), cached_read_list.end(),
                                               read, compare_reads_by_time);

            if (later_read != cached_read_list.begin()) {
                auto earlier_read = std::prev(later_read);

                if (is_within_time_and_length_criteria(*earlier_read, read)) {
                    ReadPair pair = {*earlier_read, read};
                    ++(*earlier_read)->num_duplex_candidate_pairs;
                    send_message_to_sink(std::make_shared<ReadPair>(pair));
                }
            }

            if (later_read != cached_read_list.end()) {
                if (is_within_time_and_length_criteria(read, *later_read)) {
                    ReadPair pair = {read, *later_read};
                    ++read->num_duplex_candidate_pairs;
                    send_message_to_sink(std::make_shared<ReadPair>(pair));
                }
            }

            cached_read_list.insert(later_read, read);
            while (cached_read_list.size() > m_max_num_reads) {
                cached_read_list.pop_front();
            }
        }
    }

    if (--m_num_worker_threads == 0) {
        std::unique_lock<std::mutex> lock(m_pairing_mtx);
        // There are still reads in channel_mux_read_map. Push them to the sink.
        // Last thread alive is responsible for cleaning up the cache.
        for (const auto& kv : m_channel_mux_read_map) {
            // kv is a std::pair<UniquePoreIdentifierKey, std::list<std::shared_ptr<Read>>>
            const auto& reads_list = kv.second;

            for (const auto& read_ptr : reads_list) {
                // Push each read message
                send_message_to_sink(read_ptr);
            }
        }
    }
}

PairingNode::PairingNode(std::map<std::string, std::string> template_complement_map,
                         int num_worker_threads,
                         size_t max_reads)
        : MessageSink(max_reads),
          m_num_worker_threads(num_worker_threads),
          m_template_complement_map(std::move(template_complement_map)) {
    // Set up the complement-template_map
    for (auto& key : m_template_complement_map) {
        m_complement_template_map[key.second] = key.first;
    }

    for (size_t i = 0; i < m_num_worker_threads; i++) {
        m_workers.push_back(std::make_unique<std::thread>(
                std::thread(&PairingNode::pair_list_worker_thread, this)));
    }
}

PairingNode::PairingNode(ReadOrder read_order, int num_worker_threads, size_t max_reads)
        : MessageSink(max_reads),
          m_num_worker_threads(num_worker_threads),
          m_max_num_keys(std::numeric_limits<size_t>::max()),
          m_max_num_reads(std::numeric_limits<size_t>::max()) {
    switch (read_order) {
    case ReadOrder::BY_CHANNEL:
        m_max_num_keys = 10;
        break;
    case ReadOrder::BY_TIME:
        m_max_num_reads = 10;
        break;
    default:
        throw std::runtime_error("Unsupported read order detected: " +
                                 dorado::to_string(read_order));
    }

    for (size_t i = 0; i < m_num_worker_threads; i++) {
        m_workers.push_back(std::make_unique<std::thread>(
                std::thread(&PairingNode::pair_generating_worker_thread, this)));
    }
}

void PairingNode::terminate_impl() {
    terminate_input_queue();
    for (auto& m : m_workers) {
        if (m->joinable()) {
            m->join();
        }
    }
}

stats::NamedStats PairingNode::sample_stats() const {
    stats::NamedStats stats = m_work_queue.sample_stats();
    return stats;
}

}  // namespace dorado
