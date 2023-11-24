#pragma once
#include "demux/AdapterDetector.h"
#include "read_pipeline/ReadPipeline.h"
#include "utils/stats.h"
#include "utils/types.h"

#include <atomic>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace dorado {

class AdapterDetectorNode : public MessageSink {
public:
    AdapterDetectorNode(int threads, bool trim_adapters, bool trim_primers);
    AdapterDetectorNode(int threads);
    ~AdapterDetectorNode() override;
    std::string get_name() const override { return "AdapterDetectorNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions&) override { terminate_impl(); }
    void restart() override;

private:
    void start_threads();

    size_t m_threads{1};
    bool m_trim_adapters;
    bool m_trim_primers;
    std::vector<std::unique_ptr<std::thread>> m_workers;
    std::atomic<int> m_num_records{0};
    demux::AdapterDetector m_detector;

    void worker_thread();
    void process_read(BamPtr& read);
    void process_read(SimplexRead& read);

    void terminate_impl();
};

}  // namespace dorado
