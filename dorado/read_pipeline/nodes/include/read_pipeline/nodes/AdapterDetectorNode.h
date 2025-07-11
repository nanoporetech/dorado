#pragma once

#include "demux/AdapterDetectorSelector.h"
#include "read_pipeline/base/MessageSink.h"

#include <atomic>
#include <memory>
#include <string>

namespace dorado {

namespace demux {
struct AdapterInfo;
}

class AdapterDetectorNode : public MessageSink {
public:
    AdapterDetectorNode(int threads);
    ~AdapterDetectorNode() override;

    std::string get_name() const override;
    stats::NamedStats sample_stats() const override;
    void terminate(const TerminateOptions&) override;
    void restart() override;

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
