#pragma once
#include "ReadPipeline.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace dorado {

class SubreadTaggerNode : public MessageSink {
public:
    SubreadTaggerNode(int num_worker_threads, size_t max_reads);
    ~SubreadTaggerNode() { terminate_impl(); }
    std::string get_name() const override { return "SubreadTaggerNode"; }
    ::dorado::stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions &) override { terminate_impl(); }
    void restart() override;

private:
    void start_threads();
    void terminate_impl();
    void worker_thread();
    void check_duplex_thread();

    int m_num_worker_threads = 0;
    std::vector<std::unique_ptr<std::thread>> m_worker_threads;
    std::unique_ptr<std::thread> m_duplex_thread;

    std::mutex m_subread_groups_mutex;
    std::unordered_map<uint64_t, std::vector<SimplexReadPtr>> m_subread_groups;

    std::mutex m_updated_read_tags_mutex;

    std::mutex m_duplex_reads_mutex;
    std::condition_variable m_check_duplex_cv;
    std::unordered_set<uint64_t> m_updated_read_tags;
    std::unordered_map<uint64_t, std::vector<DuplexReadPtr>> m_duplex_reads;
    std::unordered_map<uint64_t, std::pair<std::vector<SimplexReadPtr>, size_t>>
            m_full_subread_groups;

    std::atomic_bool m_terminate{false};
};

}  // namespace dorado
