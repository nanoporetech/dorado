#include "PairingNode.h"

#include "ClientInfo.h"
#include "utils/sequence_utils.h"
#include "utils/thread_naming.h"

#include <minimap.h>
#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {
const int kMaxTimeDeltaMs = 10000;
const int kMinOverlapLength = 50;
const int kMinSeqLength = 500;
const float kMinSimplexQScore = 8.f;

size_t read_signal_bytes(const dorado::SimplexRead& read) {
    return read.read_common.raw_data.nbytes();
}

// There are 4 different cases to consider when checking for adjacent reads -
// 1 Both reads are unsplit - in this case the next and prev ids determined
//     from the pod5 are unchanged and consistent.
//     i.e. temp.next == comp AND comp.prev == temp
// 2 Both reads are split from the same parent - in this case the splitter
//     adjusts the prev/next ids after splitting. The new next/prev ids
//     are also consistent within the same parent read id.
//     i.e. temp.next == comp AND comp.prev == temp
// 3 One read is split, the other is unsplit - if the split read is the template,
//     then only the template's next id will be correctly updated to the complement's id.
//     Similarly if the complement read is split, then only the complement's prev
//     id will have the template's id. So in this case only one of the pair connections
//     is correct (because during splitting only the subread's properties can be adjusted).
//     i.e. temp.next == comp OR comp.prev == temp
// 4 Both reads are split from different parents - in this case, the template read's
//     next read will point to the complement read's parent id. And vice versa for the
//     complement read's prev id.
//     i.e. temp.next == comp.parent AND comp.prev == temp.parent
bool are_reads_adjacent(const dorado::SimplexRead& temp, const dorado::SimplexRead& comp) {
    if (temp.read_common.read_id == comp.prev_read || temp.next_read == comp.read_common.read_id ||
        (temp.read_common.parent_read_id == comp.prev_read &&
         temp.next_read == comp.read_common.parent_read_id)) {
        return true;
    }
    return false;
}

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
        const dorado::SimplexRead& temp,
        const dorado::SimplexRead& comp,
        int tid) {
    if (!are_reads_adjacent(temp, comp)) {
        return {false, 0, 0, 0, 0};
    }

    int delta = int(comp.read_common.start_time_ms - temp.get_end_time_ms());
    int seq_len1 = int(temp.read_common.seq.length());
    int seq_len2 = int(comp.read_common.seq.length());
    int min_seq_len = std::min(seq_len1, seq_len2);
    int max_seq_len = std::max(seq_len1, seq_len2);
    float min_qscore = std::min(temp.read_common.calculate_mean_qscore(),
                                comp.read_common.calculate_mean_qscore());

    if ((delta < 0) || (delta >= kMaxTimeDeltaMs) || (min_seq_len < kMinSeqLength) ||
        (min_qscore < kMinSimplexQScore)) {
        return {false, 0, 0, 0, 0};
    }

    const float kEarlyAcceptSeqLenRatio = 0.98f;
    const int kEarlyAcceptTimeDeltaMs = 100;
    float len_ratio = static_cast<float>(min_seq_len) / static_cast<float>(max_seq_len);
    if (delta <= kEarlyAcceptTimeDeltaMs && len_ratio >= kEarlyAcceptSeqLenRatio &&
        min_seq_len >= 5000) {
        spdlog::trace("Early acceptance: len frac {}, delta {} temp len {}, comp len {}, {} and {}",
                      len_ratio, delta, temp.read_common.seq.length(),
                      comp.read_common.seq.length(), temp.read_common.read_id,
                      comp.read_common.read_id);
        m_early_accepted_pairs++;
        return {true, 0, int(temp.read_common.seq.length() - 1), 0,
                int(comp.read_common.seq.length() - 1)};
    }

    return is_within_alignment_criteria(temp, comp, delta, true, tid);
}

PairingNode::PairingResult PairingNode::is_within_alignment_criteria(
        const dorado::SimplexRead& temp,
        const dorado::SimplexRead& comp,
        int delta,
        bool allow_rejection,
        int tid) {
    PairingResult pair_result = {false, 0, 0, 0, 0};
    const std::string nvtx_id = "pairing_map_" + std::to_string(tid);
    nvtx3::scoped_range loop{nvtx_id};

    MmTbufPtr& working_buffer = m_tbufs[tid];
    const auto overlap_result =
            utils::compute_overlap(temp.read_common.seq, temp.read_common.read_id,
                                   comp.read_common.seq, comp.read_common.read_id, working_buffer);

    if (overlap_result) {
        const uint8_t mapq = overlap_result->mapq;
        const int32_t temp_start = overlap_result->target_start;
        const int32_t temp_end = overlap_result->target_end;
        const int32_t comp_start = overlap_result->query_start;
        const int32_t comp_end = overlap_result->query_end;
        const bool rev = overlap_result->rev;

        const int kMinMapQ = 50;
        const float kMinOverlapFraction = 0.8f;

        // Require high mapping quality.
        bool meets_mapq = (mapq >= kMinMapQ);
        // Require overlap to cover most of at least one of the reads.
        float overlap_frac =
                std::max(static_cast<float>(temp_end - temp_start) / temp.read_common.seq.length(),
                         static_cast<float>(comp_end - comp_start) / comp.read_common.seq.length());
        bool meets_length = overlap_frac > kMinOverlapFraction;
        // Require the start of the complement strand to map to end
        // of the template strand.
        bool ends_anchored = (comp_start + (temp.read_common.seq.length() - temp_end)) <= 500;
        int min_overlap_length = std::min(temp_end - temp_start, comp_end - comp_start);
        bool meets_min_overlap_length = min_overlap_length > kMinOverlapLength;
        bool cond =
                (meets_mapq && meets_length && rev && ends_anchored && meets_min_overlap_length);

        spdlog::trace(
                "mapq {}, overlap length {}, overlap frac {}, delta {}, read 1 {}, "
                "read 2 {}, strand {}, pass {}, accepted {}, temp start {} temp end {}, "
                "comp start {} comp end {}, {} and {}",
                mapq, temp_end - temp_start, overlap_frac, delta, temp.read_common.seq.length(),
                comp.read_common.seq.length(), rev ? "-" : "+", cond, !allow_rejection, temp_start,
                temp_end, comp_start, comp_end, temp.read_common.read_id, comp.read_common.read_id);

        if (cond || !allow_rejection) {
            m_overlap_accepted_pairs++;
            pair_result = {true, temp_start, temp_end, comp_start, comp_end};
        }
    }

    return pair_result;
}

void PairingNode::pair_list_worker_thread(int tid) {
    utils::set_thread_name("pair_list_thrd");
    Message message;
    while (get_input_message(message)) {
        // If this message isn't a read, just forward it to the sink.
        if (!std::holds_alternative<SimplexReadPtr>(message)) {
            send_message_to_sink(std::move(message));
            continue;
        }

        // If this message isn't a read, we'll get a bad_variant_access exception.
        auto read = std::get<SimplexReadPtr>(std::move(message));

        bool read_is_template = false;
        bool partner_found = false;
        std::string partner_id;

        // Check if read is a template with corresponding complement
        std::unique_lock<std::mutex> tc_lock(m_tc_map_mutex);

        auto it = m_template_complement_map.find(read->read_common.read_id);
        if (it != m_template_complement_map.end()) {
            partner_id = it->second;
            tc_lock.unlock();
            read_is_template = true;
            partner_found = true;
        } else {
            tc_lock.unlock();
            std::lock_guard<std::mutex> ct_lock(m_ct_map_mutex);
            it = m_complement_template_map.find(read->read_common.read_id);
            if (it != m_complement_template_map.end()) {
                partner_id = it->second;
                partner_found = true;
            }
        }

        if (partner_found) {
            std::unique_lock<std::mutex> read_cache_lock(m_read_cache_mutex);
            auto partner_read_itr = m_read_cache.find(partner_id);
            if (partner_read_itr == m_read_cache.end()) {
                // Partner is not in the read cache
                auto read_id = read->read_common.read_id;
                m_read_cache[read_id] = std::move(read);
                read_cache_lock.unlock();
            } else {
                auto partner_read = std::move(partner_read_itr->second);
                m_read_cache.erase(partner_read_itr);
                read_cache_lock.unlock();

                SimplexReadPtr template_read;
                SimplexReadPtr complement_read;

                if (read_is_template) {
                    template_read = std::move(read);
                    complement_read = std::move(partner_read);
                } else {
                    complement_read = std::move(read);
                    template_read = std::move(partner_read);
                }

                int delta = int(complement_read->read_common.start_time_ms -
                                template_read->get_end_time_ms());
                auto [is_pair, qs, qe, rs, re] = is_within_alignment_criteria(
                        *template_read, *complement_read, delta, false, tid);
                if (is_pair) {
                    ReadPair read_pair;
                    read_pair.template_read = ReadPair::ReadData::from_read(*template_read, qs, qe);
                    read_pair.complement_read =
                            ReadPair::ReadData::from_read(*complement_read, rs, re);

                    template_read->is_duplex_parent = true;
                    complement_read->is_duplex_parent = true;
                    ++template_read->num_duplex_candidate_pairs;

                    send_message_to_sink(std::move(read_pair));
                } else {
                    spdlog::debug("- rejected explicitly requested read pair: {} and {}",
                                  template_read->read_common.read_id,
                                  complement_read->read_common.read_id);
                }
            }
        }
    }
    --m_num_active_worker_threads;
}

void PairingNode::pair_generating_worker_thread(int tid) {
    utils::set_thread_name("pair_gen_thrd");
    at::InferenceMode inference_mode_guard;

    auto compare_reads_by_time = [](const SimplexReadPtr& read1, const SimplexReadPtr& read2) {
        return read1->read_common.start_time_ms < read2->read_common.start_time_ms;
    };

    Message message;
    while (get_input_message(message)) {
        if (std::holds_alternative<CacheFlushMessage>(message)) {
            std::unique_lock<std::mutex> lock(m_pairing_mtx);
            auto flush_message = std::get<CacheFlushMessage>(message);
            auto& read_cache = m_read_caches[flush_message.client_id];
            for (auto& [key, reads_list] : read_cache.channel_read_map) {
                // kv is a std::pair<UniquePoreIdentifierKey, std::list<std::shared_ptr<Read>>>
                for (auto& read_ptr : reads_list) {
                    // Push each read message
                    m_cache_signal_bytes -= read_signal_bytes(*read_ptr);
                    send_message_to_sink(std::move(read_ptr));
                }
            }
            m_read_caches.erase(flush_message.client_id);
            continue;
        }

        // If this message isn't a read, just forward it to the sink.
        if (!std::holds_alternative<SimplexReadPtr>(message)) {
            send_message_to_sink(std::move(message));
            continue;
        }

        const std::string nvtx_id = "pairing_code_" + std::to_string(tid);
        nvtx3::scoped_range loop{nvtx_id};
        // If this message isn't a read, we'll get a bad_variant_access exception.
        auto read = std::get<SimplexReadPtr>(std::move(message));

        int channel = read->read_common.attributes.channel_number;
        std::string run_id = read->read_common.run_id;
        std::string flowcell_id = read->read_common.flowcell_id;
        int32_t client_id = read->read_common.client_info->client_id();

        std::unique_lock<std::mutex> lock(m_pairing_mtx);

        auto& read_cache = m_read_caches[client_id];
        UniquePoreIdentifierKey key = std::make_tuple(channel, run_id, flowcell_id);
        auto read_list_iter = read_cache.channel_read_map.find(key);
        // Check if the key is already in the list
        if (read_list_iter == read_cache.channel_read_map.end()) {
            // Key is not in the dequeue
            // Add the new key to the end of the list
            {
                read_cache.working_channel_keys.push_back(key);
                std::list<SimplexReadPtr> reads;
                m_cache_signal_bytes += read_signal_bytes(*read);
                reads.push_back(std::move(read));
                read_cache.channel_read_map.emplace(key, std::move(reads));
            }

            if (read_cache.working_channel_keys.size() > m_max_num_keys) {
                // Remove the oldest key (front of the list)
                auto oldest_key = read_cache.working_channel_keys.front();
                read_cache.working_channel_keys.pop_front();

                auto oldest_key_it = read_cache.channel_read_map.find(oldest_key);

                // Remove the oldest key from the map
                for (auto& read_ptr : oldest_key_it->second) {
                    m_cache_signal_bytes -= read_signal_bytes(*read_ptr);
                    m_reads_to_clear.insert(std::move(read_ptr));
                }
                read_cache.channel_read_map.erase(oldest_key);
                assert(read_cache.channel_read_map.size() ==
                       read_cache.working_channel_keys.size());
            }
        } else {
            auto& cached_read_list = read_list_iter->second;
            // It's safe to take raw pointers of these reads since their ownership isn't released from this
            // node until their counter in |m_reads_in_flight_ctr| hits 0.
            SimplexRead* later_read = nullptr;
            SimplexRead* earlier_read = nullptr;

            auto later_read_iter = std::lower_bound(
                    cached_read_list.begin(), cached_read_list.end(), read, compare_reads_by_time);
            if (later_read_iter != cached_read_list.end()) {
                later_read = later_read_iter->get();
                m_reads_in_flight_ctr[later_read]++;
            }

            if (later_read_iter != cached_read_list.begin()) {
                earlier_read = std::prev(later_read_iter)->get();
                m_reads_in_flight_ctr[earlier_read]++;
            }

            SimplexRead* const read_ptr = read.get();
            m_cache_signal_bytes += read_signal_bytes(*read);
            cached_read_list.insert(later_read_iter, std::move(read));
            m_reads_in_flight_ctr[read_ptr]++;

            while (cached_read_list.size() > m_max_num_reads) {
                m_cache_signal_bytes -= read_signal_bytes(*cached_read_list.front());
                auto cached_read = std::move(cached_read_list.front());
                cached_read_list.pop_front();
                m_reads_to_clear.insert(std::move(cached_read));
            }

            // Release mutex around read cache to run pair evaluations.
            lock.unlock();

            if (later_read) {
                auto [is_pair, qs, qe, rs, re] =
                        is_within_time_and_length_criteria(*read_ptr, *later_read, tid);
                if (is_pair) {
                    ReadPair pair;
                    pair.template_read = ReadPair::ReadData::from_read(*read_ptr, qs, qe);
                    pair.complement_read = ReadPair::ReadData::from_read(*later_read, rs, re);

                    read_ptr->is_duplex_parent = true;
                    later_read->is_duplex_parent = true;
                    ++read_ptr->num_duplex_candidate_pairs;
                    send_message_to_sink(std::move(pair));
                }
            }

            if (earlier_read) {
                auto [is_pair, qs, qe, rs, re] =
                        is_within_time_and_length_criteria(*earlier_read, *read_ptr, tid);
                if (is_pair) {
                    ReadPair pair;
                    pair.template_read = ReadPair::ReadData::from_read(*earlier_read, qs, qe);
                    pair.complement_read = ReadPair::ReadData::from_read(*read_ptr, rs, re);

                    earlier_read->is_duplex_parent = true;
                    read_ptr->is_duplex_parent = true;
                    ++earlier_read->num_duplex_candidate_pairs;
                    send_message_to_sink(std::move(pair));
                }
            }

            // Acquire read cache lock again to decrement in flight read counters.
            lock.lock();

            // Decrement in-flight counter for each read.
            m_reads_in_flight_ctr[read_ptr]--;
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
            auto in_flight_itr = m_reads_in_flight_ctr.find(to_clear_itr->get());
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
            // There are still reads in channel_read_map. Push them to the sink.
            // Last thread alive is responsible for cleaning up the cache.
            for (auto& [client_id, read_cache] : m_read_caches) {
                for (auto& kv : read_cache.channel_read_map) {
                    // kv is a std::pair<UniquePoreIdentifierKey, std::list<std::shared_ptr<Read>>>
                    auto& reads_list = kv.second;

                    for (auto& read_ptr : reads_list) {
                        m_cache_signal_bytes -= read_signal_bytes(*read_ptr);
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
        : MessageSink(max_reads, 0),
          m_num_worker_threads(num_worker_threads),
          m_template_complement_map(std::move(template_complement_map)) {
    // Set up the complement-template_map
    for (auto& key : m_template_complement_map) {
        m_complement_template_map[key.second] = key.first;
    }

    m_pairing_func = &PairingNode::pair_list_worker_thread;
}

PairingNode::PairingNode(DuplexPairingParameters pairing_params,
                         int num_worker_threads,
                         size_t max_reads)
        : MessageSink(max_reads, 0),
          m_num_worker_threads(num_worker_threads),
          m_max_num_keys(std::numeric_limits<size_t>::max()),
          m_max_num_reads(std::numeric_limits<size_t>::max()) {
    switch (pairing_params.read_order) {
    case ReadOrder::BY_CHANNEL:
        // N.B. with BY_CHANNEL ordering the ont_basecall_client application has a dependency
        // on how the the cache is structured, i.e. that the number of channels in the cache
        // is set to pairing_params.cache_depth. This is so that it can calculate the theoretical
        // max cache size given it knows the max number of reads per channel that it will send.
        // If the way the cache is structured is changed the ont_basecall_client code will also
        // need to be updated otherwise there is a risk of deadlock.
        m_max_num_keys = pairing_params.cache_depth;
        spdlog::debug("Using dorado duplex channel count of {}", m_max_num_keys);
        break;
    case ReadOrder::BY_TIME:
        m_max_num_reads = pairing_params.cache_depth;
        spdlog::debug("Using dorado duplex read-per-channel count of {}", m_max_num_reads);
        break;
    default:
        throw std::runtime_error("Unsupported read order detected: " +
                                 dorado::to_string(pairing_params.read_order));
    }
    m_pairing_func = &PairingNode::pair_generating_worker_thread;
}

void PairingNode::start_threads() {
    m_tbufs.reserve(m_num_worker_threads);
    for (int i = 0; i < m_num_worker_threads; i++) {
        m_tbufs.push_back(MmTbufPtr(mm_tbuf_init()));
        m_workers.emplace_back([=] { (this->*m_pairing_func)(i); });
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
        m.join();
    }
    m_workers.clear();

    m_tbufs.clear();
}

void PairingNode::restart() {
    start_input_queue();
    start_threads();
}

stats::NamedStats PairingNode::sample_stats() const {
    stats::NamedStats stats = m_work_queue.sample_stats();
    stats["early_accepted_pairs"] = m_early_accepted_pairs.load();
    stats["overlap_accepted_pairs"] = m_overlap_accepted_pairs.load();
    stats["cached_signal_mb"] =
            static_cast<double>(m_cache_signal_bytes) / static_cast<double>(1024 * 1024);
    return stats;
}

}  // namespace dorado
