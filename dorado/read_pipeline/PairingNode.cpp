#include "PairingNode.h"

#include "minimap.h"

#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {
const int kMaxTimeDeltaMs = 1000;
const float kMinSeqLenRatio = 0.2f;
const int kMinOverlapLength = 50;
}  // namespace

namespace dorado {

// Determine whether 2 proposed reads form a duplex pair or not.
// The algorithm utilizes the following heuristics to make a decision -
// 1. Reads must be within 1000ms of each other, and the ratio of their
//    lengths must be at least 20%.
// 2. If the lengths are >98% similar, reads are at least 5KB, and time
//    delta is <100ms, consider them to be a pair.
// 3. If the early acceptance fails, then run minimap2 to generate overlap
//    coordinates. If there is only 1 hit from minimap2 mapping,
//    the mapping quality is high (>50), the overlap covers
//    most of the shorter read (80%), the overlap is at least 50 bp long,
//    one read maps to the reverse strand of the other, and the end
//    of the template is mapped to the beginning
//    of the complement read, then consider them a pair.
PairingNode::PairingResult PairingNode::is_within_time_and_length_criteria(
        const std::shared_ptr<dorado::Read>& temp,
        const std::shared_ptr<dorado::Read>& comp,
        int tid) {
    int delta = comp->start_time_ms - temp->get_end_time_ms();
    int seq_len1 = temp->seq.length();
    int seq_len2 = comp->seq.length();
    int min_seq_len = std::min(seq_len1, seq_len2);
    int max_seq_len = std::max(seq_len1, seq_len2);
    float len_ratio = static_cast<float>(min_seq_len) / static_cast<float>(max_seq_len);

    if ((delta < 0) || (delta >= kMaxTimeDeltaMs) || (len_ratio <= kMinSeqLenRatio) ||
        (min_seq_len <= kMinOverlapLength)) {
        return {false, 0, 0, 0, 0};
    }

    const float kEarlyAcceptSeqLenRatio = 0.98;
    const int kEarlyAcceptTimeDeltaMs = 100;
    if (delta <= kEarlyAcceptTimeDeltaMs && len_ratio >= kEarlyAcceptSeqLenRatio &&
        min_seq_len >= 5000) {
        spdlog::debug("Early acceptance: len frac {}, delta {} temp len {}, comp len {}, {} and {}",
                      len_ratio, delta, temp->seq.length(), comp->seq.length(), temp->read_id,
                      comp->read_id);
        m_early_accepted_pairs++;
        return {true, 0, temp->seq.length() - 1, 0, comp->seq.length() - 1};
    }

    return is_within_alignment_criteria(temp, comp, delta, true, tid);
}

PairingNode::PairingResult PairingNode::is_within_alignment_criteria(
        const std::shared_ptr<dorado::Read>& temp,
        const std::shared_ptr<dorado::Read>& comp,
        int delta,
        bool allow_rejection,
        int tid) {
    PairingResult pair_result = {false, 0, 0, 0, 0};
    const std::string nvtx_id = "pairing_map_" + std::to_string(tid);
    nvtx3::scoped_range loop{nvtx_id};
    // Add mm2 based overlap check.
    mm_idxopt_t m_idx_opt;
    mm_mapopt_t m_map_opt;
    mm_set_opt(0, &m_idx_opt, &m_map_opt);
    mm_set_opt("map-hifi", &m_idx_opt, &m_map_opt);

    std::vector<const char*> seqs = {temp->seq.c_str()};
    std::vector<const char*> names = {temp->read_id.c_str()};
    mm_idx_t* m_index = mm_idx_str(m_idx_opt.w, m_idx_opt.k, 0, m_idx_opt.bucket_bits, 1,
                                   seqs.data(), names.data());
    mm_mapopt_update(&m_map_opt, m_index);

    mm_tbuf_t* tbuf = m_tbufs[tid].get();

    int hits = 0;
    mm_reg1_t* reg = mm_map(m_index, comp->seq.length(), comp->seq.c_str(), &hits, tbuf, &m_map_opt,
                            comp->read_id.c_str());

    mm_idx_destroy(m_index);

    // Multiple hits implies ambiguous mapping, so ignore those pairs.
    if (hits == 1 || (!allow_rejection && hits > 0)) {
        uint8_t mapq = 0;
        int32_t temp_start = 0;
        int32_t temp_end = 0;
        int32_t comp_start = 0;
        int32_t comp_end = 0;
        bool rev = false;

        auto best_map = &reg[0];
        mapq = best_map->mapq;
        temp_start = best_map->rs;
        temp_end = best_map->re;
        comp_start = best_map->qs;
        comp_end = best_map->qe;
        rev = best_map->rev;

        const int kMinMapQ = 50;
        const float kMinOverlapFraction = 0.8f;

        // Require high mapping quality.
        bool meets_mapq = (mapq >= kMinMapQ);
        // Require overlap to cover most of at least one of the reads.
        float overlap_frac =
                std::max(static_cast<float>(temp_end - temp_start) / temp->seq.length(),
                         static_cast<float>(comp_end - comp_start) / comp->seq.length());
        bool meets_length = overlap_frac > kMinOverlapFraction;
        // Require the start of the complement strand to map to end
        // of the template strand.
        bool ends_anchored = (comp_start + (temp->seq.length() - temp_end)) <= 500;
        int min_overlap_length = std::min(temp_end - temp_start, comp_end - comp_start);
        bool meets_min_overlap_length = min_overlap_length > kMinOverlapLength;
        bool cond =
                (meets_mapq && meets_length && rev && ends_anchored && meets_min_overlap_length);

        spdlog::debug(
                "hits {}, mapq {}, overlap length {}, overlap frac {}, delta {}, read 1 {}, "
                "read 2 {}, strand {}, pass {}, accepted {}, temp start {} temp end {}, "
                "comp start {} comp end {}, {} and {}",
                hits, mapq, temp_end - temp_start, overlap_frac, delta, temp->seq.length(),
                comp->seq.length(), rev ? "-" : "+", cond, !allow_rejection, temp_start, temp_end,
                comp_start, comp_end, temp->read_id, comp->read_id);

        if (cond || !allow_rejection) {
            m_overlap_accepted_pairs++;
            pair_result = {true, temp_start, temp_end, comp_start, comp_end};
        }
    }

    for (int i = 0; i < hits; ++i) {
        free(reg[i].p);
    }
    free(reg);

    return pair_result;
}

void PairingNode::pair_list_worker_thread(int tid) {
    Message message;
    while (get_input_message(message)) {
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

                int delta = complement_read->start_time_ms - template_read->get_end_time_ms();
                auto [is_pair, qs, qe, rs, re] = is_within_alignment_criteria(
                        template_read, complement_read, delta, false, tid);
                if (is_pair) {
                    ReadPair read_pair = {template_read, complement_read, qs, qe, rs, re};
                    template_read->is_duplex_parent = true;
                    complement_read->is_duplex_parent = true;

                    ++template_read->num_duplex_candidate_pairs;

                    send_message_to_sink(std::make_shared<ReadPair>(read_pair));
                } else {
                    spdlog::debug("- rejected explicitly requested read pair: {} and {}",
                                  template_read->read_id, complement_read->read_id);
                }
            }
        }
    }
    --m_num_active_worker_threads;
}

void PairingNode::pair_generating_worker_thread(int tid) {
    torch::InferenceMode inference_mode_guard;

    auto compare_reads_by_time = [](const std::shared_ptr<Read>& read1,
                                    const std::shared_ptr<Read>& read2) {
        return read1->start_time_ms < read2->start_time_ms;
    };

    Message message;
    while (get_input_message(message)) {
        if (std::holds_alternative<CacheFlushMessage>(message)) {
            std::unique_lock<std::mutex> lock(m_pairing_mtx);
            auto flush_message = std::get<CacheFlushMessage>(message);
            auto& read_cache = m_read_caches[flush_message.client_id];
            for (auto& [key, reads_list] : read_cache.channel_mux_read_map) {
                // kv is a std::pair<UniquePoreIdentifierKey, std::list<std::shared_ptr<Read>>>
                for (auto& read_ptr : reads_list) {
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
                    m_reads_to_clear.insert(std::move(read_ptr));
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
                m_reads_in_flight_ctr[later_read]++;
            }

            if (later_read_iter != cached_read_list.begin()) {
                earlier_read = *(std::prev(later_read_iter));
                m_reads_in_flight_ctr[earlier_read]++;
            }

            cached_read_list.insert(later_read_iter, read);
            m_reads_in_flight_ctr[read]++;

            while (cached_read_list.size() > m_max_num_reads) {
                auto cached_read = cached_read_list.front();
                cached_read_list.pop_front();
                m_reads_to_clear.insert(std::move(cached_read));
            }

            // Release mutex around read cache to run pair evaluations.
            lock.unlock();

            bool found_pair = false;
            if (later_read) {
                auto [is_pair, qs, qe, rs, re] =
                        is_within_time_and_length_criteria(read, later_read, tid);
                if (is_pair) {
                    ReadPair pair = {read, later_read, qs, qe, rs, re};
                    read->is_duplex_parent = true;
                    later_read->is_duplex_parent = true;
                    ++read->num_duplex_candidate_pairs;
                    send_message_to_sink(std::make_shared<ReadPair>(pair));
                    found_pair = true;
                }
            }

            if (!found_pair && earlier_read) {
                auto [is_pair, qs, qe, rs, re] =
                        is_within_time_and_length_criteria(earlier_read, read, tid);
                if (is_pair) {
                    ReadPair pair = {earlier_read, read, qs, qe, rs, re};
                    earlier_read->is_duplex_parent = true;
                    read->is_duplex_parent = true;
                    ++(earlier_read)->num_duplex_candidate_pairs;
                    send_message_to_sink(std::make_shared<ReadPair>(pair));
                }
            }

            // Acquire read cache lock again to decrement in flight read counters.
            lock.lock();

            // Decrement in-flight counter for each read.
            m_reads_in_flight_ctr[read]--;
            if (earlier_read) {
                m_reads_in_flight_ctr[earlier_read]--;
            }
            if (later_read) {
                m_reads_in_flight_ctr[later_read]--;
            }
        }

        // Once pairs have been evaluated, check if any of the in-flight reads
        // need to be purged from the cache.
        for (auto to_clear_itr = m_reads_to_clear.begin();
             to_clear_itr != m_reads_to_clear.end();) {
            auto in_flight_itr = m_reads_in_flight_ctr.find(*to_clear_itr);
            bool ok_to_clear = false;
            // If a read to clear is not in-flight (not in the in-flight list
            // or in-flight counter is 0), then clear it
            // from the cache.
            if (in_flight_itr == m_reads_in_flight_ctr.end()) {
                ok_to_clear = true;
            } else if (in_flight_itr->second.load() == 0) {
                m_reads_in_flight_ctr.erase(in_flight_itr);
                ok_to_clear = true;
            }
            if (ok_to_clear) {
                auto read_handle = m_reads_to_clear.extract(*to_clear_itr++);
                send_message_to_sink(std::move(read_handle.value()));
            } else {
                ++to_clear_itr;
            }
        }
    }

    if (--m_num_active_worker_threads == 0) {
        if (!m_preserve_cache_during_flush) {
            std::unique_lock<std::mutex> lock(m_pairing_mtx);
            // There are still reads in channel_mux_read_map. Push them to the sink.
            // Last thread alive is responsible for cleaning up the cache.
            for (auto& [client_id, read_cache] : m_read_caches) {
                for (auto& kv : read_cache.channel_mux_read_map) {
                    // kv is a std::pair<UniquePoreIdentifierKey, std::list<std::shared_ptr<Read>>>
                    auto& reads_list = kv.second;

                    for (auto& read_ptr : reads_list) {
                        // Push each read message
                        send_message_to_sink(std::move(read_ptr));
                    }
                }
            }
            m_read_caches.clear();
        }
        m_reads_in_flight_ctr.clear();
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

    m_pairing_func = &PairingNode::pair_list_worker_thread;
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
    m_pairing_func = &PairingNode::pair_generating_worker_thread;
    start_threads();
}

void PairingNode::start_threads() {
    m_tbufs.reserve(m_num_worker_threads);
    for (size_t i = 0; i < m_num_worker_threads; i++) {
        m_tbufs.push_back(MmTbufPtr(mm_tbuf_init()));
        m_workers.push_back(std::make_unique<std::thread>(std::thread(m_pairing_func, this, i)));
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

    m_tbufs.clear();
}

void PairingNode::restart() {
    restart_input_queue();
    start_threads();
}

stats::NamedStats PairingNode::sample_stats() const {
    stats::NamedStats stats = m_work_queue.sample_stats();
    stats["early_accepted_pairs"] = m_early_accepted_pairs.load();
    stats["overlap_accepted_pairs"] = m_overlap_accepted_pairs.load();
    return stats;
}

}  // namespace dorado
