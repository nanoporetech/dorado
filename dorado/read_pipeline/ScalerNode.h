#pragma once
#include "ReadPipeline.h"

class ScalerNode : public ReadSink {
public: 
    ScalerNode(ReadSink& sink, size_t max_reads=1000);
    ~ScalerNode();
    int trim(torch::Tensor signal, int window_size=40, float threshold_factor=2.4, int min_elements=3);

private:
    void worker_thread();
    ReadSink& m_sink;
    std::unique_ptr<std::thread> m_worker;
};
