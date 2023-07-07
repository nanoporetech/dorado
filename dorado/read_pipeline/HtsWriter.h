#pragma once
#include "htslib/sam.h"
#include "read_pipeline/ReadPipeline.h"
#include "utils/stats.h"

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

    HtsWriter(const std::string& filename, OutputMode mode, size_t threads, size_t num_reads);
    ~HtsWriter();
    std::string get_name() const override { return "HtsWriter"; }
    stats::NamedStats sample_stats() const override;
    void terminate() override { terminate_impl(); }

    int set_and_write_header(const sam_hdr_t* header);
    static OutputMode get_output_mode(const std::string& mode);
    size_t get_total() const { return m_total; }
    size_t get_primary() const { return m_primary; }
    size_t get_unmapped() const { return m_unmapped; }

private:
    void terminate_impl();
    size_t m_total{0};
    size_t m_primary{0};
    size_t m_unmapped{0};
    size_t m_secondary{0};
    size_t m_supplementary{0};
    sam_hdr_t* m_header{nullptr};

    htsFile* m_file{nullptr};
    std::unique_ptr<std::thread> m_worker;
    void worker_thread();
    int write(bam1_t* record);
    size_t m_num_reads_expected;
    std::unordered_set<std::string> m_processed_read_ids;
};

}  // namespace dorado
