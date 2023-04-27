#pragma once
#include "../nn/ModelRunner.h"
#include "ReadPipeline.h"

namespace dorado {

class StereoDuplexEncoderNode : public MessageSink {
public:
    // Chunk size and overlap are in raw samples
    StereoDuplexEncoderNode(MessageSink &sink,
                            std::map<std::string, std::string> template_complement_map,
                            int input_signal_stride);

    std::shared_ptr<dorado::Read> stereo_encode(std::shared_ptr<dorado::Read> template_read,
                                                std::shared_ptr<dorado::Read> complement_read);

    ~StereoDuplexEncoderNode();

private:
    // Consume reads from input queue
    void worker_thread();
    MessageSink &m_sink;

    std::mutex m_tc_map_mutex;
    std::mutex m_ct_map_mutex;
    std::mutex m_read_cache_mutex;

    std::vector<std::unique_ptr<std::thread>> worker_threads;
    std::atomic<int> m_num_worker_threads;

    // Time when Basecaller Node is initialised. Used for benchmarking and debugging
    std::chrono::time_point<std::chrono::system_clock> initialization_time;

    // Time when Basecaller Node terminates. Used for benchmarking and debugging
    std::chrono::time_point<std::chrono::system_clock> termination_time;

    std::map<std::string, std::string> m_template_complement_map;
    std::map<std::string, std::string> m_complement_template_map;
    std::map<std::string, std::shared_ptr<Read>> read_cache;

    // The stride which was used to simplex call the data
    int m_input_signal_stride;
};

}  // namespace dorado