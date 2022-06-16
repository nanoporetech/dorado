#pragma once
#include "ReadPipeline.h"

class ModBaseCallerNode : public ReadSink {
public:
    ModBaseCallerNode(ReadSink& sink, size_t max_reads = 1000);
    ~ModBaseCallerNode();

private:
    void worker_thread();  // Worker thread performs calling asynchronously.

    ReadSink& m_sink;
    std::unique_ptr<std::thread> m_worker;
};