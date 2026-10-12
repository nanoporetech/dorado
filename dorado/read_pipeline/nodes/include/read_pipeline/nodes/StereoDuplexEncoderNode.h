#pragma once

#include "read_pipeline/base/MessageSink.h"

#include <atomic>
#include <cstdint>

namespace dorado {

DuplexReadPtr stereo_encode(ReadPair pair, int input_signal_stride);

class StereoDuplexEncoderNode final : public MessageSink {
public:
    explicit StereoDuplexEncoderNode(int input_signal_stride);
    ~StereoDuplexEncoderNode();

    std::string get_name() const override;
    stats::NamedStats sample_stats() const override;
    void terminate(const TerminateOptions&) override;
    void restart() override;

private:
    StereoDuplexEncoderNode(const StereoDuplexEncoderNode&) = delete;
    StereoDuplexEncoderNode& operator=(const StereoDuplexEncoderNode&) = delete;
    StereoDuplexEncoderNode(StereoDuplexEncoderNode&&) = delete;
    StereoDuplexEncoderNode& operator=(StereoDuplexEncoderNode&&) = delete;

private:
    void input_thread_fn();

    // The stride which was used to simplex call the data
    const int m_input_signal_stride;

    // Performance monitoring stats.
    std::atomic<int64_t> m_num_encoded_pairs{0};
};

}  // namespace dorado
