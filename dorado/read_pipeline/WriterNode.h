#pragma once

#include "ReadPipeline.h"

#include <atomic>
#include <string>
#include <vector>

class WriterNode : public ReadSink {
public:
    // Writer has no sink - reads go to output
    WriterNode(std::vector<std::string> args,
               bool emit_fastq,
               size_t num_worker_threads = 1,
               size_t max_reads = 1000);
    ~WriterNode();

private:
    void worker_thread();

    void print_header();

    std::vector<std::string> m_args;
    // Emit Fastq if true
    bool m_emit_fastq;
    // Total number of raw samples from the read WriterNode has processed. Used for performance benchmarking and debugging.
    std::atomic<int64_t> m_num_samples_processed;
    //Total number of reads WriterNode has processed
    std::atomic<int> m_num_reads_processed;
    // Time when Node is initialised.
    std::chrono::time_point<std::chrono::system_clock> m_initialization_time;
    // Async worker for writing.
    std::vector<std::unique_ptr<std::thread>> m_workers;

    std::mutex m_cout_mutex;
    std::mutex m_cerr_mutex;
};
