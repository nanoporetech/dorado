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
    return (delta < max_time_delta_ms) && len_ratio >= min_seq_len_ratio;
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
            if (read_cache.find(partner_id) == read_cache.end()) {
                // Partner is not in the read cache
                read_cache[read->read_id] = read;
                read_cache_lock.unlock();
            } else {
                auto partner_read_itr = read_cache.find(partner_id);
                auto partner_read = partner_read_itr->second;
                read_cache.erase(partner_read_itr);
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

                m_sink.push_message(std::make_shared<ReadPair>(read_pair));
            }
        }
    }
    if (--m_num_worker_threads == 0) {
        m_sink.terminate();
    }
}

void PairingNode::pair_generating_worker_thread() {
    Message message;
    while (m_work_queue.try_pop(message)) {
        // If this message isn't a read, we'll get a bad_variant_access exception.
        auto read = std::get<std::shared_ptr<Read>>(message);

        int channel = read->attributes.channel_number;
        int mux = read->attributes.mux;
        std::string run_id = read->run_id;
        std::string flowcell_id = read->flowcell_id;

        int max_num_keys = 10;
        std::unique_lock<std::mutex> lock(m_pairing_mtx);
        UniquePoreIdentifierKey key = std::make_tuple(channel, mux, run_id, flowcell_id);
        auto found = channel_mux_read_map.find(key);
        // Check if the key is already in the list
        if (found == channel_mux_read_map.end()) {
            // Key is not in the dequeue

            if (m_working_channel_mux_keys.size() >= max_num_keys) {
                // Remove the oldest key (front of the list)
                auto oldest_key = m_working_channel_mux_keys.front();
                m_working_channel_mux_keys.pop_front();

                auto oldest_key_it = channel_mux_read_map.find(oldest_key);

                // Remove the oldest key from the map
                for (auto read_ptr : oldest_key_it->second) {
                    m_sink.push_message(read_ptr);
                }
                channel_mux_read_map.erase(oldest_key);
                assert(channel_mux_read_map.size() == m_working_channel_mux_keys.size());
            }
            // Add the new key to the end of the list
            m_working_channel_mux_keys.push_back(key);
        }

        auto compare_reads_by_time = [](const std::shared_ptr<Read>& read1,
                                        const std::shared_ptr<Read>& read2) {
            return read1->attributes.start_time < read2->attributes.start_time;
        };

        if (channel_mux_read_map.count(key)) {
            auto later_read =
                    std::lower_bound(channel_mux_read_map[key].begin(),
                                     channel_mux_read_map[key].end(), read, compare_reads_by_time);

            if (later_read != channel_mux_read_map[key].begin()) {
                auto earlier_read = std::prev(later_read);

                if (is_within_time_and_length_criteria(*earlier_read, read)) {
                    ReadPair pair = {*earlier_read, read};
                    m_sink.push_message(std::make_shared<ReadPair>(pair));
                }
            }

            if (later_read != channel_mux_read_map[key].end()) {
                if (is_within_time_and_length_criteria(read, *later_read)) {
                    ReadPair pair = {read, *later_read};
                    m_sink.push_message(std::make_shared<ReadPair>(pair));
                }
            }

            channel_mux_read_map[key].insert(later_read, read);

        } else {
            channel_mux_read_map[key].push_back(read);
        }
    }
    if (--m_num_worker_threads == 0) {
        std::unique_lock<std::mutex> lock(m_pairing_mtx);
        // There are still reads in channel_mux_read_map. Push them to the sink.
        // Last thread alive is responsible for cleaning up the cache.
        for (const auto& kv : channel_mux_read_map) {
            // kv is a std::pair<UniquePoreIdentifierKey, std::list<std::shared_ptr<Read>>>
            const auto& reads_list = kv.second;

            for (const auto& read_ptr : reads_list) {
                // Push each read message
                m_sink.push_message(read_ptr);
            }
        }

        m_sink.terminate();
    }
}

PairingNode::PairingNode(MessageSink& sink,
                         std::optional<std::map<std::string, std::string>> template_complement_map)
        : m_sink(sink), MessageSink(1000), m_num_worker_threads(2) {
    if (template_complement_map.has_value()) {
        m_template_complement_map = template_complement_map.value();
        // Set up the complement-template_map
        for (auto& key : m_template_complement_map) {
            m_complement_template_map[key.second] = key.first;
        }

        for (size_t i = 0; i < m_num_worker_threads; i++) {
            m_workers.push_back(std::make_unique<std::thread>(
                    std::thread(&PairingNode::pair_list_worker_thread, this)));
        }
    } else {
        for (size_t i = 0; i < m_num_worker_threads; i++) {
            m_workers.push_back(std::make_unique<std::thread>(
                    std::thread(&PairingNode::pair_generating_worker_thread, this)));
        }
    }
}

PairingNode::~PairingNode() {
    terminate();
    for (auto& m : m_workers) {
        m->join();
    }
}
}  // namespace dorado
