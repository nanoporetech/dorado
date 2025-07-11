#pragma once

#include "read_pipeline/base/MessageSink.h"

#include <atomic>
#include <cstdint>

namespace dorado {

class StereoDuplexEncoderNode : public MessageSink {
public:
    StereoDuplexEncoderNode(int input_signal_stride);
    ~StereoDuplexEncoderNode();

    std::string get_name() const override;
    stats::NamedStats sample_stats() const override;
    void terminate(const TerminateOptions &) override;
    void restart() override;

    DuplexReadPtr stereo_encode(ReadPair pair);

private:
    void input_thread_fn();

    // The stride which was used to simplex call the data
    const int m_input_signal_stride;

    // Performance monitoring stats.
    std::atomic<int64_t> m_num_encoded_pairs{0};
};

}  // namespace dorado
