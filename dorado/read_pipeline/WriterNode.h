#pragma once

#include <string>
#include <vector>

#include "ReadPipeline.h"

class WriterNode : public ReadSink {
public:
    // Writer has no sink - reads go to output
    WriterNode(std::vector<std::string> args, bool emit_fastq = false, size_t max_reads=1000);
    ~WriterNode();
private:
    void worker_thread();

    std::vector<std::string> m_args;
    bool m_emit_fastq;
    int m_num_samples_processed;
    int m_num_reads_processed;
    std::chrono::time_point<std::chrono::system_clock> m_initialization_time;
    std::unique_ptr<std::thread> m_worker;
};
