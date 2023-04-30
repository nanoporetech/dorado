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
    PairingNode(MessageSink& sink, std::map<std::string, std::string> template_complement_map);
    ~PairingNode();

private:
    void worker_thread();
    std::vector<std::unique_ptr<std::thread>> m_workers;
    MessageSink& m_sink;
    std::map<std::string, std::string> m_template_complement_map;
    std::map<std::string, std::string> m_complement_template_map;

    std::mutex m_tc_map_mutex;
    std::mutex m_ct_map_mutex;
    std::mutex m_read_cache_mutex;

    std::atomic<int> m_num_worker_threads;

    std::map<std::string, std::shared_ptr<Read>> read_cache;
};

}  // namespace dorado
