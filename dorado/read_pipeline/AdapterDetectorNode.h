#pragma once

#include "demux/AdapterDetectorSelector.h"
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
    AdapterDetectorNode(int threads,
                        bool trim_adapters,
                        bool trim_primers,
                        const std::optional<std::string>& custom_seqs);
    AdapterDetectorNode(int threads);
    ~AdapterDetectorNode() override { stop_input_processing(); }
    std::string get_name() const override { return "AdapterDetectorNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions&) override { stop_input_processing(); }
    void restart() override { start_input_processing(&AdapterDetectorNode::input_thread_fn, this); }

private:
    std::shared_ptr<const AdapterInfo> m_default_adapter_info;
    std::vector<std::unique_ptr<std::thread>> m_workers;
    std::atomic<int> m_num_records{0};
    demux::AdapterDetectorSelector m_detector_selector{};

    void input_thread_fn();
    void process_read(BamPtr& read);
    void process_read(SimplexRead& read);
};

}  // namespace dorado
