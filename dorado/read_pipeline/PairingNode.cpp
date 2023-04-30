#include "PairingNode.h"

namespace dorado {

void PairingNode::worker_thread() {
    Message message;
    while (m_work_queue.try_pop(message)) {
        // If this message isn't a read, we'll get a bad_variant_access exception.
        auto read = std::get<std::shared_ptr<Read>>(message);

        bool read_is_template = false;
        bool partner_found = false;
        std::string partner_id;

        // Check if read is a template with corresponding complement
        std::unique_lock<std::mutex> tc_lock(m_tc_map_mutex);

        if (m_template_complement_map.find(read->read_id) != m_template_complement_map.end()) {
            partner_id = m_template_complement_map[read->read_id];
            tc_lock.unlock();
            read_is_template = true;
            partner_found = true;
        } else {
            tc_lock.unlock();
            std::unique_lock<std::mutex> ct_lock(m_ct_map_mutex);
            if (m_complement_template_map.find(read->read_id) != m_complement_template_map.end()) {
                partner_id = m_complement_template_map[read->read_id];
                partner_found = true;
            }
            ct_lock.unlock();
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
    int num_worker_threads = --m_num_worker_threads;
    if (num_worker_threads == 0) {
        m_sink.terminate();
    }
}

PairingNode::PairingNode(MessageSink& sink,
                         std::map<std::string, std::string> template_complement_map)
        : m_sink(sink),
          MessageSink(1000),
          m_num_worker_threads(2),
          m_template_complement_map(template_complement_map) {
    // Set up the complement-template_map
    for (auto key : template_complement_map) {
        m_complement_template_map[key.second] = key.first;
    }

    int num_worker_threads = m_num_worker_threads;
    for (size_t i = 0; i < num_worker_threads; i++) {
        m_workers.push_back(
                std::make_unique<std::thread>(std::thread(&PairingNode::worker_thread, this)));
    }
}

PairingNode::~PairingNode() {
    terminate();
    for (auto& m : m_workers) {
        m->join();
    }
}
}  // namespace dorado
