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

    // The stride which was used to simplex call the data
    int m_input_signal_stride;
};

}  // namespace dorado