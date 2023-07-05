#pragma once
#include "ReadPipeline.h"

#include <atomic>
#include <list>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace dorado {

class SubreadTaggerNode : public MessageSink {
public:
    SubreadTaggerNode(int num_worker_threads = 1, size_t max_reads = 1000);
    ~SubreadTaggerNode() { terminate_impl(); }
    void terminate() override { terminate_impl(); }

private:
    void terminate_impl();
    void worker_thread();

    std::vector<std::unique_ptr<std::thread>> worker_threads;

    std::mutex m_subread_groups_mutex;
    std::unordered_map<uint64_t, std::vector<std::shared_ptr<Read>>> m_subread_groups;

    std::mutex m_duplex_reads_mutex;
    std::list<std::shared_ptr<Read>> m_duplex_reads;
    std::list<std::vector<std::shared_ptr<Read>>> m_full_subread_groups;
};

}  // namespace dorado
