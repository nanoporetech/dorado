#pragma once

#include "ReadPipeline.h"
#include "utils/stats.h"

#include <atomic>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace dorado {

class PairingNode : public MessageSink {
public:
    enum class ReadOrder { pore_order = 0, time_order };

    // Template-complement map: uses the pair_list pairing method
    PairingNode(MessageSink& sink,
                std::map<std::string, std::string> template_complement_map,
                int num_worker_threads = 2,
                size_t max_reads = 1000);

    // No template-complement map: uses the pair_generation pairing method
    PairingNode(MessageSink& sink,
                ReadOrder read_order = ReadOrder::pore_order,
                int num_worker_threads = 2,
                size_t max_reads = 1000);
    ~PairingNode();
    std::string get_name() const override { return "PairingNode"; }
    stats::NamedStats sample_stats() const override;

private:
    void pair_list_worker_thread();
    void pair_generating_worker_thread(size_t max_num_keys, size_t max_num_reads);

    // A key for a unique Pore, Duplex reads must have the same UniquePoreIdentifierKey
    // The values are channel, mux, run_id, flowcell_id, client_id
    using UniquePoreIdentifierKey = std::tuple<int, int, std::string, std::string, int32_t>;

    MessageSink& m_sink;
    std::vector<std::unique_ptr<std::thread>> m_workers;
    std::atomic<int> m_num_worker_threads;

    // Members for pair_list method

    std::mutex m_tc_map_mutex;
    std::mutex m_ct_map_mutex;
    std::mutex m_read_cache_mutex;

    std::map<std::string, std::string> m_template_complement_map;
    std::map<std::string, std::string> m_complement_template_map;
    std::map<std::string, std::shared_ptr<Read>> m_read_cache;

    // Members for pair_generating method

    std::mutex m_pairing_mtx;

    std::map<UniquePoreIdentifierKey, std::list<std::shared_ptr<Read>>> m_channel_mux_read_map;
    std::deque<UniquePoreIdentifierKey> m_working_channel_mux_keys;
};

}  // namespace dorado
