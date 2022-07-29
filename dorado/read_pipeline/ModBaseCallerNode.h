#pragma once
#include "ReadPipeline.h"

#include <memory>

class RemoraRunner;

class ModBaseCallerNode : public ReadSink {
public:
    ModBaseCallerNode(ReadSink& sink,
                      std::shared_ptr<RemoraRunner> model_runner,
                      size_t model_stride,
                      size_t max_reads = 1000);
    ~ModBaseCallerNode();

private:
    void worker_thread();  // Worker thread performs calling asynchronously.

    ReadSink& m_sink;
    std::shared_ptr<RemoraRunner> m_model_runner;
    size_t m_model_stride;  // stride of the basecall model that was used to process the read
    std::unique_ptr<std::thread> m_worker;
};