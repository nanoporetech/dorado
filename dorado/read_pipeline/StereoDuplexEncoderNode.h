#pragma once
#include "ReadPipeline.h"
#include "utils/stats.h"

#include <atomic>
#include <memory>
#include <vector>

namespace dorado {

class StereoDuplexEncoderNode : public MessageSink {
public:
    StereoDuplexEncoderNode(int input_signal_stride);

    DuplexReadPtr stereo_encode(const ReadPair& pair);

    ~StereoDuplexEncoderNode() { terminate_impl(); };
    std::string get_name() const override { return "StereoDuplexEncoderNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions&) override { terminate_impl(); }
    void restart() override;

private:
    void start_threads();
    void terminate_impl();
    // Consume reads from input queue
    void worker_thread();

    std::vector<std::unique_ptr<std::thread>> m_worker_threads;

    // The stride which was used to simplex call the data
    int m_input_signal_stride;

    // Performance monitoring stats.
    std::atomic<int64_t> m_num_encoded_pairs = 0;
};

}  // namespace dorado
