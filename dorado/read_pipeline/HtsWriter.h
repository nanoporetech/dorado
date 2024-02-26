#pragma once
#include "read_pipeline/ReadPipeline.h"
#include "utils/stats.h"

#include <htslib/sam.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <unordered_set>

namespace dorado {

class HtsWriter : public MessageSink {
public:
    enum class OutputMode {
        UBAM,
        BAM,
        SAM,
        FASTQ,
    };

    HtsWriter(const std::string& filename, OutputMode mode, size_t threads);
    ~HtsWriter();
    std::string get_name() const override { return "HtsWriter"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions&) override { stop_input_processing(); }
    void restart() override { start_input_processing(&HtsWriter::input_thread_fn, this); }

    int set_and_write_header(const sam_hdr_t* header);
    static OutputMode get_output_mode(const std::string& mode);
    size_t get_total() const { return m_total; }
    size_t get_primary() const { return m_primary; }
    size_t get_unmapped() const { return m_unmapped; }

private:
    size_t m_total{0};
    size_t m_primary{0};
    size_t m_unmapped{0};
    size_t m_secondary{0};
    size_t m_supplementary{0};

    class HtsFile;
    std::unique_ptr<HtsFile> m_file;

    void input_thread_fn();
    int write(bam1_t* record);
    std::unordered_set<std::string> m_processed_read_ids;
    std::atomic<int> m_duplex_reads_written{0};
    std::atomic<int> m_split_reads_written{0};
};

}  // namespace dorado
