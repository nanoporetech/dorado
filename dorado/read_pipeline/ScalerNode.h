#pragma once
#include "ReadPipeline.h"

namespace dorado {

class ScalerNode : public MessageSink {
public:
    ScalerNode(MessageSink& sink, int num_worker_threads = 5, size_t max_reads = 1000);
    ~ScalerNode();

private:
    void worker_thread();  // Worker thread performs scaling and trimming asynchronously.
    MessageSink&
            m_sink;  // MessageSink to consume scaled reads. Typically this will be a Basecaller Node.
    std::vector<std::unique_ptr<std::thread>> worker_threads;
    std::atomic<int> m_num_worker_threads;
};

}  // namespace dorado
