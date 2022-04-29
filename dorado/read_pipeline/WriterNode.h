#pragma once
#include "ReadPipeline.h"

class WriterNode : public ReadSink {
public:
    WriterNode(bool emit_same = false, size_t max_reads=1000); // Writer has no sink - reads go to output
    ~WriterNode();
private:
    void worker_thread();

    bool m_emit_sam;
    int m_num_samples_processed;
    int m_num_reads_processed;
    std::chrono::time_point<std::chrono::system_clock> m_initialization_time;
    std::unique_ptr<std::thread> m_worker;
};
