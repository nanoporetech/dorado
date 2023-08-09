#pragma once

#include "ReadPipeline.h"
#include "utils/stats.h"
#include "utils/types.h"

#include <atomic>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace dorado {

class PairingNode : public MessageSink {
    // A key for a unique Pore, Duplex reads must have the same UniquePoreIdentifierKey
    // The values are channel, mux, run_id, flowcell_id
    using UniquePoreIdentifierKey = std::tuple<int, int, std::string, std::string>;

    struct ReadCache {
        std::map<UniquePoreIdentifierKey, std::list<std::shared_ptr<Read>>> channel_mux_read_map;
        std::deque<UniquePoreIdentifierKey> working_channel_mux_keys;
    };

public:
    // Template-complement map: uses the pair_list pairing method
    PairingNode(std::map<std::string, std::string> template_complement_map,
                int num_worker_threads = 2,
                size_t max_reads = 1000);

    // No template-complement map: uses the pair_generation pairing method
    PairingNode(ReadOrder read_order, int num_worker_threads = 2, size_t max_reads = 1000);
    ~PairingNode() { terminate_impl(); }
    std::string get_name() const override { return "PairingNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions& flush_options) override;
    void restart() override;

private:
    void start_threads();
    void terminate_impl();

    /**
     * This is a worker thread function for pairing reads based on a specified list of template-complement pairs.
     */
    void pair_list_worker_thread(int tid);

    /**
     * This is a worker thread function for generating pairs of reads that fall within pairing criteria.
     * 
     * The function goes through the incoming messages, which are expected to be reads. For each read, it finds its pore 
     * in the list of active pores. If the pore isn't in the list yet, it is added. If the list of active pores has reached 
     * its maximum size (m_max_num_keys), the oldest pore is removed from the list, and its associated reads are discarded.
     * The function then inserts the new read into the sorted list of reads for its pore, and checks if it can be paired 
     * with the reads immediately before and after it in the list. If the list of reads for a pore has reached its maximum 
     * size (m_max_num_reads), the oldest read is removed from the list.
     */
    void pair_generating_worker_thread(int tid);

    std::vector<std::unique_ptr<std::thread>> m_workers;
    int m_num_worker_threads = 0;
    std::atomic<int> m_num_active_worker_threads = 0;
    std::atomic<bool> m_preserve_cache_during_flush = false;

    using FPairingFunc = void (PairingNode::*)(int);
    FPairingFunc m_pairing_func = nullptr;

    // Members for pair_list method

    std::mutex m_tc_map_mutex;
    std::mutex m_ct_map_mutex;
    std::mutex m_read_cache_mutex;

    std::map<std::string, std::string> m_template_complement_map;
    std::map<std::string, std::string> m_complement_template_map;
    std::map<std::string, std::shared_ptr<Read>> m_read_cache;

    // Members for pair_generating method

    std::mutex m_pairing_mtx;

    // individual read caches per client, keyed by client_id
    std::unordered_map<int32_t, ReadCache> m_read_caches;

    /**
     * The maximum number of different channels (pores) to keep in memory concurrently. 
     * This parameter is crucial when reads are expected to be delivered in channel/pore order. In this order, 
     * once a read from a specific pore is processed, it is guaranteed that no other reads from that pore will appear.
     * Thus, the function can limit memory usage by only keeping reads from a fixed number of pores (channels) in memory.
     */
    size_t m_max_num_keys;

    /**
     * The maximum number of reads from a specific pore to keep in memory. This parameter is 
     * crucial when reads are expected to be delivered in time order. In this order, reads from the same pore could 
     * appear at any point in the stream. Thus, the function keeps a limited history of reads for each pore in memory.
     * It ensures that the memory usage is controlled, while the reads needed for pairing are available.    
     */
    size_t m_max_num_reads;

    using PairingResult = std::tuple<bool, uint32_t, uint32_t, uint32_t, uint32_t>;
    PairingResult is_within_time_and_length_criteria(const std::shared_ptr<dorado::Read>& read1,
                                                     const std::shared_ptr<dorado::Read>& read2,
                                                     int tid);

    PairingResult is_within_alignment_criteria(const std::shared_ptr<dorado::Read>& temp,
                                               const std::shared_ptr<dorado::Read>& comp,
                                               int delta,
                                               bool allow_rejection,
                                               int tid);

    // Store the minimap2 buffers used for mapping. One buffer per thread.
    std::vector<MmTbufPtr> m_tbufs;

    // Track reads which need to be emptied from the cache but are still being
    // evaluated for pairs by other threads.
    std::unordered_map<std::shared_ptr<Read>, std::atomic<int>> m_reads_in_flight_ctr;
    std::unordered_set<std::shared_ptr<Read>> m_reads_to_clear;

    // Stats tracking for pairing node.
    std::atomic<int> m_early_accepted_pairs{0};
    std::atomic<int> m_overlap_accepted_pairs{0};
};

}  // namespace dorado
