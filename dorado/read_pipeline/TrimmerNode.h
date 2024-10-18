#pragma once

#include "MessageSink.h"
#include "utils/stats.h"

#include <atomic>
#include <memory>
#include <string>
#include <vector>

namespace dorado {

class TrimmerNode : public MessageSink {
public:
    TrimmerNode(int threads);
    ~TrimmerNode() override { stop_input_processing(); }
    std::string get_name() const override { return "TrimmerNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions&) override { stop_input_processing(); }
    void restart() override {
        start_input_processing([this] { input_thread_fn(); }, "trimmer");
    }

private:
    std::atomic<int> m_num_records{0};

    void input_thread_fn();
    void process_read(BamMessage& bam_message);
    void process_read(SimplexRead& read);
};

}  // namespace dorado
