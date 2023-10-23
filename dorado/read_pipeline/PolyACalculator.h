#pragma once

#include "ReadPipeline.h"
#include "utils/stats.h"

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace dorado {

class PolyACalculator : public MessageSink {
public:
    enum ModelType {
        DNA,
        RNA002,
        RNA004,
    };

    PolyACalculator(size_t num_worker_threads, ModelType model_type, size_t max_reads = 1000);
    ~PolyACalculator() { terminate_impl(); }
    std::string get_name() const override { return "PolyACalculator"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions& flush_options) override { terminate_impl(); };
    void restart() override;
    static ModelType get_model_type(const std::string& model_name);

private:
    void start_threads();
    void terminate_impl();
    void worker_thread();

    // Async worker for writing.
    std::vector<std::unique_ptr<std::thread>> m_workers;
    size_t m_num_worker_threads = 0;
    const bool m_is_rna;
    const ModelType m_model_type;
    std::atomic<size_t> total_tail_lengths_called{0};
    std::atomic<int> num_called{0};
    std::atomic<int> num_not_called{0};

    std::mutex m_mutex;
    std::map<int, int> tail_length_counts;
};

}  // namespace dorado
