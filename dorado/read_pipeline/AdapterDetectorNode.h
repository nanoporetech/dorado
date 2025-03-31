#pragma once

#include "MessageSink.h"
#include "demux/AdapterDetectorSelector.h"
#include "utils/stats.h"

#include <atomic>
#include <memory>
#include <string>
#include <vector>

namespace dorado {

namespace demux {
struct AdapterInfo;
}

class AdapterDetectorNode : public MessageSink {
public:
    AdapterDetectorNode(int threads);
    ~AdapterDetectorNode() override { stop_input_processing(); }
    std::string get_name() const override { return "AdapterDetectorNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions&) override { stop_input_processing(); }
    void restart() override {
        start_input_processing([this] { input_thread_fn(); }, "adapter_detect");
    }

private:
    std::atomic<int> m_num_records{0};
    std::atomic<int> m_num_untrimmed_short_reads{0};
    demux::AdapterDetectorSelector m_detector_selector{};

    void input_thread_fn();
    void process_read(BamMessage& bam_message);
    void process_read(SimplexRead& read);
    std::shared_ptr<demux::AdapterDetector> get_detector(const demux::AdapterInfo& adapter_info);
};

}  // namespace dorado
