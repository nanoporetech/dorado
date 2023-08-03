#pragma once
#include "htslib/sam.h"
#include "read_pipeline/ReadPipeline.h"
#include "utils/stats.h"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>

namespace dorado {

class BarcodeDemuxer : public MessageSink {
public:
    BarcodeDemuxer(const std::string& output_dir,
                   size_t threads,
                   size_t num_reads,
                   bool write_fastq);
    ~BarcodeDemuxer();
    std::string get_name() const override { return "BarcodeDemuxer"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions& flush_options) override { terminate_impl(); }
    void restart() override;

    void set_header(const sam_hdr_t* header);

private:
    void terminate_impl();
    void start_threads();
    std::filesystem::path m_output_dir;
    int m_threads;
    size_t m_total{0};
    size_t m_primary{0};
    size_t m_unmapped{0};
    size_t m_secondary{0};
    size_t m_supplementary{0};
    sam_hdr_t* m_header{nullptr};
    std::atomic<int> m_processed_reads{0};

    std::unordered_map<std::string, htsFile*> m_files;
    std::unique_ptr<std::thread> m_worker;
    void worker_thread();
    int write(bam1_t* record);
    size_t m_num_reads_expected;
    bool m_write_fastq{false};
};

}  // namespace dorado
