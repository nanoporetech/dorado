#pragma once

#include "ReadPipeline.h"
#include "data_loader/DataLoader.h"

#include <indicators/progress_bar.hpp>

#include <atomic>
#include <string>
#include <vector>

namespace dorado {

class WriterNode : public ReadSink {
public:
    // Writer has no sink - reads go to output
    WriterNode(std::vector<std::string> args,
               bool emit_fastq,
               bool emit_moves,
               bool rna,
               bool duplex,
               size_t min_qscore,
               size_t num_worker_threads = 1,
               std::unordered_map<std::string, ReadGroup> = {},
               int num_reads = 0,
               size_t max_reads = 1000);
    ~WriterNode();

private:
    void worker_thread();

    void print_header();

    std::vector<std::string> m_args;
    size_t m_min_qscore;
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
    //Total number of reads with a mean qscore less the m_min_qscore
    std::atomic<int> m_num_reads_failed;
    // Time when Node is initialised.
    std::chrono::time_point<std::chrono::system_clock> m_initialization_time;
    // Async worker for writing.
    std::vector<std::unique_ptr<std::thread>> m_workers;
    std::mutex m_cout_mutex;
    std::mutex m_cerr_mutex;
    int m_progress_bar_increment;

    // Progress bar for showing basecalling progress
    indicators::ProgressBar m_progress_bar{
            indicators::option::Stream{std::cerr},     indicators::option::BarWidth{30},
            indicators::option::ShowElapsedTime{true}, indicators::option::ShowRemainingTime{true},
            indicators::option::ShowPercentage{true},
    };
};

}  // namespace dorado
