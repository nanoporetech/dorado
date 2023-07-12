#include "PairingNode.h"

#include "minimap.h"

#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace dorado {

std::tuple<bool, uint32_t, uint32_t, uint32_t, uint32_t>
PairingNode::is_within_time_and_length_criteria(const std::shared_ptr<dorado::Read>& read1,
                                                const std::shared_ptr<dorado::Read>& read2,
                                                int tid) {
    const int max_time_delta_ms = 1000;
    int delta = read2->start_time_ms - read1->get_end_time_ms();
    int seq_len1 = read1->seq.length();
    int seq_len2 = read2->seq.length();
    float len_ratio = static_cast<float>(std::min(seq_len1, seq_len2)) /
                      static_cast<float>(std::max(seq_len1, seq_len2));
    float min_seq_len_ratio = 0.2f;

    if ((delta >= 0) && (delta < max_time_delta_ms) && (len_ratio > min_seq_len_ratio)) {
        float accept_seq_len_ratio = 0.98;
        if (delta <= 100 && len_ratio >= accept_seq_len_ratio) {
            spdlog::debug("Early acceptance: len frac {}, delta {} read1 len {}, read2 len {}",
                          len_ratio, delta, read1->seq.length(), read2->seq.length());
            return {true, 0, read1->seq.length() - 1, 0, read2->seq.length() - 1};
        }

        //spdlog::info("{} tid {} mm2 start", std::string(tid, '\t'), tid);
        const std::string nvtx_id = "pairing_mm2_" + std::to_string(tid);
        nvtx3::scoped_range loop{nvtx_id};
        // Add mm2 based overlap check.
        mm_idxopt_t m_idx_opt;
        mm_mapopt_t m_map_opt;
        mm_set_opt(0, &m_idx_opt, &m_map_opt);
        mm_set_opt("map-hifi", &m_idx_opt, &m_map_opt);

        std::vector<const char*> seqs = {read1->seq.c_str()};
        std::vector<const char*> names = {read1->read_id.c_str()};
        //spdlog::info("{} tid {} index start", std::string(tid, '\t'), tid);
        mm_idx_t* m_index = mm_idx_str(m_idx_opt.w, m_idx_opt.k, 0, m_idx_opt.bucket_bits, 1,
                                       seqs.data(), names.data());
        //spdlog::info("{} tid {} index end", std::string(tid, '\t'), tid);
        mm_mapopt_update(&m_map_opt, m_index);

        //mm_tbuf_t* tbuf = mm_tbuf_init();
        mm_tbuf_t* tbuf = m_tbufs[tid];

        int hits = 0;
        //spdlog::info("{} tid {} map start", std::string(tid, '\t'), tid);
        mm_reg1_t* reg = mm_map(m_index, read2->seq.length(), read2->seq.c_str(), &hits, tbuf,
                                &m_map_opt, read2->read_id.c_str());
        //spdlog::info("{} tid {} map end ({} to {})", std::string(tid, '\t'), tid, read2->seq.length(), read1->seq.length());

        uint8_t mapq = 0;
        int32_t r1s = 0;
        int32_t r1e = 0;
        int32_t r2s = 0;
        int32_t r2e = 0;
        bool rev = false;
        if (hits > 0) {
            auto best_map =
                    std::max_element(reg, reg + hits, [&](const mm_reg1_t& a, const mm_reg1_t& b) {
                        return std::abs(a.qe - a.qs) < std::abs(b.qe - b.qs);
                    });
            mapq = best_map->mapq;
            r1s = best_map->rs;
            r1e = best_map->re;
            r2s = best_map->qs;
            r2e = best_map->qe;
            rev = best_map->rev;

            free(best_map->p);
        }

        bool meets_mapq = (mapq >= 50);
        float overlap_frac = std::max(static_cast<float>(r1e - r1s) / read1->seq.length(),
                                      static_cast<float>(r2e - r2s) / read2->seq.length());
        bool meets_length = overlap_frac > 0.8f;
        bool cond = (meets_mapq && meets_length && rev);

        //mm_tbuf_destroy(tbuf);
        mm_idx_destroy(m_index);

        spdlog::debug(
                "hits {}, mapq {}, overlap length {}, overlap frac {}, delta {}, read 1 {}, read 2 "
                "{}, strand {}, pass {}, qs {} qe {}, rs {} re {}, {} and {}",
                hits, mapq, r1e - r1s, overlap_frac, delta, read1->seq.length(),
                read2->seq.length(), rev ? "-" : "+", cond, r1s, r1e, r2s, r2e, read1->read_id,
                read2->read_id);

        //spdlog::info("{} tid {} mm2 end", std::string(tid, '\t'), tid);
        if (cond) {
            return {true, r1s, r1e, r2s, r2e};
        }
    }
    return {false, 0, 0, 0, 0};
}

void PairingNode::pair_list_worker_thread() {
    Message message;
    while (m_work_queue.try_pop(message)) {
        // If this message isn't a read, just forward it to the sink.
        if (!std::holds_alternative<std::shared_ptr<Read>>(message)) {
            send_message_to_sink(std::move(message));
            continue;
        }

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
    --m_num_active_worker_threads;
}

void PairingNode::pair_generating_worker_thread(int tid) {
    auto compare_reads_by_time = [](const std::shared_ptr<Read>& read1,
                                    const std::shared_ptr<Read>& read2) {
        return read1->start_time_ms < read2->start_time_ms;
    };

    Message message;
    while (m_work_queue.try_pop(message)) {
        if (std::holds_alternative<CacheFlushMessage>(message)) {
            std::unique_lock<std::mutex> lock(m_pairing_mtx);
            auto flush_message = std::get<CacheFlushMessage>(message);
            auto& read_cache = m_read_caches[flush_message.client_id];
            for (const auto& [key, reads_list] : read_cache.channel_mux_read_map) {
                // kv is a std::pair<UniquePoreIdentifierKey, std::list<std::shared_ptr<Read>>>
                for (const auto& read_ptr : reads_list) {
                    // Push each read message
                    send_message_to_sink(std::move(read_ptr));
                }
            }
            m_read_caches.erase(flush_message.client_id);
            continue;
        }

        // If this message isn't a read, just forward it to the sink.
        if (!std::holds_alternative<std::shared_ptr<Read>>(message)) {
            send_message_to_sink(std::move(message));
            continue;
        }

        // If this message isn't a read, we'll get a bad_variant_access exception.
        const std::string nvtx_id = "pairing_code_" + std::to_string(tid);
        nvtx3::scoped_range loop{nvtx_id};
        auto read = std::get<std::shared_ptr<Read>>(message);
        //spdlog::info("{} tid {} pop read", std::string(tid, '\t'), tid);

        int channel = read->attributes.channel_number;
        int mux = read->attributes.mux;
        std::string run_id = read->run_id;
        std::string flowcell_id = read->flowcell_id;
        int32_t client_id = read->client_id;

        std::unique_lock<std::mutex> lock(m_pairing_mtx);

        auto& read_cache = m_read_caches[client_id];
        UniquePoreIdentifierKey key = std::make_tuple(channel, mux, run_id, flowcell_id);
        auto read_list_iter = read_cache.channel_mux_read_map.find(key);
        // Check if the key is already in the list
        if (read_list_iter == read_cache.channel_mux_read_map.end()) {
            // Key is not in the dequeue
            // Add the new key to the end of the list
            read_cache.working_channel_mux_keys.push_back(key);
            read_cache.channel_mux_read_map.insert({key, {read}});

            if (read_cache.working_channel_mux_keys.size() > m_max_num_keys) {
                // Remove the oldest key (front of the list)
                auto oldest_key = read_cache.working_channel_mux_keys.front();
                read_cache.working_channel_mux_keys.pop_front();

                auto oldest_key_it = read_cache.channel_mux_read_map.find(oldest_key);

                // Remove the oldest key from the map
                for (auto read_ptr : oldest_key_it->second) {
                    send_message_to_sink(read_ptr);
                }
                read_cache.channel_mux_read_map.erase(oldest_key);
                assert(read_cache.channel_mux_read_map.size() ==
                       read_cache.working_channel_mux_keys.size());
            }
        } else {
            auto& cached_read_list = read_list_iter->second;
            std::shared_ptr<Read> later_read, earlier_read;
            auto later_read_iter = std::lower_bound(
                    cached_read_list.begin(), cached_read_list.end(), read, compare_reads_by_time);
            if (later_read_iter != cached_read_list.end()) {
                later_read = *later_read_iter;
            }

            if (later_read_iter != cached_read_list.begin()) {
                earlier_read = *(std::prev(later_read_iter));
            }

            cached_read_list.insert(later_read_iter, read);
            while (cached_read_list.size() > m_max_num_reads) {
                auto cached_read = cached_read_list.front();
                cached_read_list.pop_front();
                send_message_to_sink(std::move(cached_read));
            }

            lock.unlock();  // Release mutex around read cache.

            //spdlog::info("{} tid {} check start", std::string(tid, '\t'), tid);
            if (later_read) {
                auto [is_pair, qs, qe, rs, re] =
                        is_within_time_and_length_criteria(read, later_read, tid);
                if (is_pair) {
                    ReadPair pair = {read, later_read, qs, qe, rs, re};
                    read->is_duplex_parent = true;
                    later_read->is_duplex_parent = true;
                    ++read->num_duplex_candidate_pairs;
                    send_message_to_sink(std::make_shared<ReadPair>(pair));
                    continue;
                }
            }

            if (earlier_read) {
                auto [is_pair, qs, qe, rs, re] =
                        is_within_time_and_length_criteria(earlier_read, read, tid);
                if (is_pair) {
                    ReadPair pair = {earlier_read, read, qs, qe, rs, re};
                    earlier_read->is_duplex_parent = true;
                    read->is_duplex_parent = true;
                    ++(earlier_read)->num_duplex_candidate_pairs;
                    send_message_to_sink(std::make_shared<ReadPair>(pair));
                    continue;
                }
            }
        }
        //spdlog::info("{} tid {} done", std::string(tid, '\t'), tid);
    }

    if (--m_num_active_worker_threads == 0) {
        if (!m_preserve_cache_during_flush) {
            std::unique_lock<std::mutex> lock(m_pairing_mtx);
            //spdlog::info("Push leftover reads to encoder");
            // There are still reads in channel_mux_read_map. Push them to the sink.
            // Last thread alive is responsible for cleaning up the cache.
            for (const auto& [client_id, read_cache] : m_read_caches) {
                for (const auto& kv : read_cache.channel_mux_read_map) {
                    // kv is a std::pair<UniquePoreIdentifierKey, std::list<std::shared_ptr<Read>>>
                    const auto& reads_list = kv.second;

                    for (const auto& read_ptr : reads_list) {
                        // Push each read message
                        send_message_to_sink(read_ptr);
                    }
                }
            }
            m_read_caches.clear();
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

    start_threads();
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
    start_threads();
}

void PairingNode::start_threads() {
    for (size_t i = 0; i < m_num_worker_threads; i++) {
        m_tbufs.push_back(mm_tbuf_init());
        m_workers.push_back(std::make_unique<std::thread>(
                std::thread(&PairingNode::pair_generating_worker_thread, this, i)));
        ++m_num_active_worker_threads;
    }
}

void PairingNode::terminate(const FlushOptions& flush_options) {
    m_preserve_cache_during_flush = flush_options.preserve_pairing_caches;
    terminate_impl();
    m_preserve_cache_during_flush = false;
}

void PairingNode::terminate_impl() {
    terminate_input_queue();
    for (auto& m : m_workers) {
        if (m->joinable()) {
            m->join();
        }
    }
    m_workers.clear();

    for (int i = 0; i < m_tbufs.size(); i++) {
        mm_tbuf_destroy(m_tbufs[i]);
    }
    m_tbufs.clear();
}

void PairingNode::restart() {
    restart_input_queue();
    start_threads();
}

stats::NamedStats PairingNode::sample_stats() const {
    stats::NamedStats stats = m_work_queue.sample_stats();
    return stats;
}

}  // namespace dorado
