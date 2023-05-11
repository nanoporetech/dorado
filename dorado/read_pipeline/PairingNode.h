#pragma once

#include "ReadPipeline.h"

#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace dorado {

class PairingNode : public MessageSink {
public:
    PairingNode(MessageSink& sink,
                std::optional<std::map<std::string, std::string>> = std::nullopt);
    ~PairingNode();

private:
    void pair_list_worker_thread();
    void pair_generating_worker_thread();

    // A key for a unique Pore, Duplex reads must have the same UniquePoreIdentifierKey
    // The values are channel, mux, run_id, flowcell_id
    using UniquePoreIdentifierKey = std::tuple<int, int, std::string, std::string>;

    std::vector<std::unique_ptr<std::thread>> m_workers;
    MessageSink& m_sink;
    std::map<std::string, std::string> m_template_complement_map;
    std::map<std::string, std::string> m_complement_template_map;

    std::mutex m_tc_map_mutex;
    std::mutex m_ct_map_mutex;
    std::mutex m_read_cache_mutex;

    std::atomic<int> m_num_worker_threads;

    std::map<std::string, std::shared_ptr<Read>> read_cache;

    std::map<UniquePoreIdentifierKey, std::list<std::shared_ptr<Read>>> channel_mux_read_map;

    std::deque<UniquePoreIdentifierKey> m_working_channel_mux_keys;

    std::mutex m_pairing_mtx;
};

}  // namespace dorado
