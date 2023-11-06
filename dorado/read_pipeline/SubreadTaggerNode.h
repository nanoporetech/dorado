#pragma once
#include "ReadPipeline.h"

#include <atomic>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dorado {

class SubreadTaggerNode : public MessageSink {
public:
    SubreadTaggerNode(int num_worker_threads = 1, size_t max_reads = 1000);
    ~SubreadTaggerNode() { terminate_impl(); }
    std::string get_name() const override { return "SubreadTaggerNode"; }
    ::dorado::stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions& flush_options) override { terminate_impl(); }
    void restart() override;

private:
    void start_threads();
    void terminate_impl();
    void worker_thread();

    int m_num_worker_threads = 0;
    std::vector<std::unique_ptr<std::thread>> m_worker_threads;

    std::mutex m_subread_groups_mutex;
    std::unordered_map<uint64_t, std::vector<SimplexReadPtr>> m_subread_groups;

    std::mutex m_duplex_reads_mutex;
    std::list<DuplexReadPtr> m_duplex_reads;
    std::list<std::pair<std::vector<SimplexReadPtr>, std::vector<DuplexReadPtr>>>
            m_full_subread_groups;
};

}  // namespace dorado
