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

    std::vector<std::unique_ptr<std::thread>> m_workers;
    MessageSink& m_sink;
    std::map<std::string, std::string> m_template_complement_map;
    std::map<std::string, std::string> m_complement_template_map;

    std::mutex m_tc_map_mutex;
    std::mutex m_ct_map_mutex;
    std::mutex m_read_cache_mutex;

    std::atomic<int> m_num_worker_threads;

    std::map<std::string, std::shared_ptr<Read>> read_cache;

    std::map<std::tuple<int, int, std::string, std::string>, std::list<std::shared_ptr<Read>>>
            channel_mux_read_map;
    std::atomic<int> read_counter = 0;
    std::mutex
            m_channel_mux_read_map_mtx;  //TODO: Need to santiy check if this is thread-safe, should be static

    std::deque<std::tuple<int, int, std::string, std::string>> m_working_channel_mux_keys;
    std::mutex m_working_channel_mux_key_list_mtx;
};

}  // namespace dorado
