#pragma once
#include "ReadPipeline.h"

class WriterNode : public ReadSink {
public:
    WriterNode(size_t max_reads=1000); // Writer has no sink - reads go to output
    ~WriterNode();
private:
    void worker_thread();
    std::unique_ptr<std::thread> m_worker;
    int num_samples_processed;
    int num_reads_processed;
    std::chrono::time_point<std::chrono::system_clock> initialization_time;
};
