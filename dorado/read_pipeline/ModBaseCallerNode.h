#pragma once
#include "ReadPipeline.h"

#include <memory>

class RemoraRunner;

class ModBaseCallerNode : public ReadSink {
public:
    ModBaseCallerNode(ReadSink& sink,
                      std::vector<std::shared_ptr<RemoraRunner>>& model_runners,
                      size_t max_reads = 1000);
    ~ModBaseCallerNode();

private:
    void worker_thread();  // Worker thread performs calling asynchronously.

    ReadSink& m_sink;
    std::unique_ptr<std::thread> m_worker;
};