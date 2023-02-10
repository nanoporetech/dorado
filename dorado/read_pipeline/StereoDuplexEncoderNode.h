#pragma once
#include "../nn/ModelRunner.h"
#include "ReadPipeline.h"

namespace dorado {

class StereoDuplexEncoderNode : public ReadSink {
public:
    // Chunk size and overlap are in raw samples
    StereoDuplexEncoderNode(ReadSink &sink,
                            std::map<std::string, std::string> template_complement_map);
    ~StereoDuplexEncoderNode();

private:
    // Consume reads from input queue
    void worker_thread();
    ReadSink &m_sink;

    std::mutex m_tc_map_mutex;
    std::mutex m_ct_map_mutex;
    std::mutex m_read_cache_mutex;

    std::vector<std::unique_ptr<std::thread>> worker_threads;

    // Time when Basecaller Node is initialised. Used for benchmarking and debugging
    std::chrono::time_point<std::chrono::system_clock> initialization_time;

    // Time when Basecaller Node terminates. Used for benchmarking and debugging
    std::chrono::time_point<std::chrono::system_clock> termination_time;

    std::map<std::string, std::string> m_template_complement_map;
    std::map<std::string, std::string> m_complement_template_map;
    std::map<std::string, std::shared_ptr<Read>> read_cache;
};

}  // namespace dorado