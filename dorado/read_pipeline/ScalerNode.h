#pragma once
#include "ReadPipeline.h"

class ScalerNode : public ReadSink {
public:
    ScalerNode(ReadSink& sink, size_t max_reads = 1000);
    ~ScalerNode();
    // Read Trimming method (removes some initial part of the raw read).
    int trim(torch::Tensor signal,
             int window_size = 40,
             float threshold_factor = 2.4,
             int min_elements = 3);

private:
    void worker_thread();  // Worker thread performs scaling and trimming asynchronously.
    ReadSink&
            m_sink;  // ReadSink to consume scaled reads. Typically this will be a Basecaller Node.
    std::unique_ptr<std::thread> m_worker;
};