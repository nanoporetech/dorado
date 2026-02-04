#pragma once

#include "read_pipeline/base/MessageSink.h"

#include <atomic>
#include <string>

namespace dorado {

class TrimmerNode : public MessageSink {
public:
    TrimmerNode(int threads, bool is_rna);
    ~TrimmerNode() override;

    std::string get_name() const override;
    stats::NamedStats sample_stats() const override;
    void terminate(const TerminateOptions&) override;
    void restart() override;

private:
    std::atomic<int> m_num_records{0};
    const bool m_is_rna;

    void input_thread_fn();
    void process_read(BamMessage& bam_message);
    void process_read(SimplexRead& read);
};

}  // namespace dorado
