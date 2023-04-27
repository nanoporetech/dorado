#pragma once

#include "ReadPipeline.h"
#include "utils/types.h"

#include <indicators/progress_bar.hpp>

#include <atomic>
#include <string>
#include <vector>

struct sam_hdr_t;
struct htsFile;

namespace dorado {

class WriterNode : public MessageSink {
public:
    // Writer has no sink - reads go to output
    WriterNode(std::vector<std::string> args,
               bool emit_fastq,
               bool emit_moves,
               bool rna,
               bool duplex,
               size_t num_worker_threads = 1,
               std::unordered_map<std::string, ReadGroup> = {},
               int num_reads = 0,
               size_t max_reads = 1000);
    ~WriterNode();

private:
    void worker_thread();

    void add_header();

    std::vector<std::string> m_args;
    // Read Groups - print these in header.
    std::unordered_map<std::string, ReadGroup> m_read_groups;
    // Emit Fastq if true
    bool m_emit_fastq, m_emit_moves, m_isatty, m_duplex, m_rna;
    // Total number of raw samples from the read WriterNode has processed. Used for performance benchmarking and debugging.
    std::atomic<int64_t> m_num_bases_processed;
    std::atomic<int64_t> m_num_samples_processed;
    //Total number of reads WriterNode has processed
    std::atomic<int> m_num_reads_processed;
    //Total number of reads WriterNode expects to process
    std::atomic<int> m_num_reads_expected;
    // Time when Node is initialised.
    std::chrono::time_point<std::chrono::system_clock> m_initialization_time;
    // Async worker for writing.
    std::vector<std::unique_ptr<std::thread>> m_workers;
    std::atomic<size_t> m_active_threads;
    std::mutex m_cout_mutex;
    std::mutex m_cerr_mutex;
    int m_progress_bar_increment;

    // Progress bar for showing basecalling progress
    indicators::ProgressBar m_progress_bar{
            indicators::option::Stream{std::cerr},     indicators::option::BarWidth{30},
            indicators::option::ShowElapsedTime{true}, indicators::option::ShowRemainingTime{true},
            indicators::option::ShowPercentage{true},
    };

    sam_hdr_t* m_header{nullptr};
    htsFile* m_file{nullptr};
};

}  // namespace dorado
