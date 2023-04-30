#pragma once
#include "../nn/ModelRunner.h"
#include "ReadPipeline.h"

namespace dorado {

class StereoDuplexEncoderNode : public MessageSink {
public:
    StereoDuplexEncoderNode(MessageSink &sink, int input_signal_stride);

    std::shared_ptr<dorado::Read> stereo_encode(std::shared_ptr<dorado::Read> template_read,
                                                std::shared_ptr<dorado::Read> complement_read);

    ~StereoDuplexEncoderNode();

private:
    // Consume reads from input queue
    void worker_thread();
    MessageSink &m_sink;

    std::vector<std::unique_ptr<std::thread>> worker_threads;
    std::atomic<int> m_num_worker_threads;

    // Time when Basecaller Node is initialised. Used for benchmarking and debugging
    std::chrono::time_point<std::chrono::system_clock> initialization_time;

    // Time when Basecaller Node terminates. Used for benchmarking and debugging
    std::chrono::time_point<std::chrono::system_clock> termination_time;

    // The stride which was used to simplex call the data
    int m_input_signal_stride;
};

}  // namespace dorado