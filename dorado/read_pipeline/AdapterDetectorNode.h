#pragma once

#include "demux/AdapterDetector.h"
#include "read_pipeline/MessageSink.h"
#include "utils/stats.h"
#include "utils/types.h"

#include <atomic>
#include <memory>
#include <string>
#include <vector>

namespace dorado {

class AdapterDetectorNode : public MessageSink {
public:
    AdapterDetectorNode(int threads, bool trim_adapters, bool trim_primers);
    AdapterDetectorNode(int threads);
    ~AdapterDetectorNode() override { stop_input_processing(); }
    std::string get_name() const override { return "AdapterDetectorNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions&) override { stop_input_processing(); }
    void restart() override { start_input_processing(&AdapterDetectorNode::input_thread_fn, this); }

private:
    bool m_trim_adapters;
    bool m_trim_primers;
    std::vector<std::unique_ptr<std::thread>> m_workers;
    std::atomic<int> m_num_records{0};
    demux::AdapterDetector m_detector;

    void input_thread_fn();
    void process_read(BamPtr& read);
    void process_read(SimplexRead& read);
    static void check_and_update_barcoding(SimplexRead& read, std::pair<int, int>& trim_interval);
};

}  // namespace dorado
