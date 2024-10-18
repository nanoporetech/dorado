#pragma once

#include "MessageSink.h"
#include "utils/stats.h"

#include <atomic>
#include <cstdint>

namespace dorado {

class StereoDuplexEncoderNode : public MessageSink {
public:
    StereoDuplexEncoderNode(int input_signal_stride);

    DuplexReadPtr stereo_encode(ReadPair pair);

    ~StereoDuplexEncoderNode() { stop_input_processing(); };
    std::string get_name() const override { return "StereoDuplexEncoderNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions &) override { stop_input_processing(); }
    void restart() override {
        start_input_processing([this] { input_thread_fn(); }, "stereo_encode");
    }

private:
    void input_thread_fn();

    // The stride which was used to simplex call the data
    const int m_input_signal_stride;

    // Performance monitoring stats.
    std::atomic<int64_t> m_num_encoded_pairs{0};
};

}  // namespace dorado
