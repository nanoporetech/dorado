#pragma once
#include "htslib/sam.h"
#include "read_pipeline/ReadPipeline.h"
#include "utils/stats.h"

#ifdef WIN32
#include <indicators/progress_bar.hpp>
#else
#include <indicators/block_progress_bar.hpp>
#endif

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>

namespace dorado {

class HtsWriter : public MessageSink {
public:
    enum OutputMode {
        UBAM,
        BAM,
        SAM,
        FASTQ,
    };

    HtsWriter(const std::string& filename, OutputMode mode, size_t threads, size_t num_reads, const sam_hdr_t* header);
    ~HtsWriter();
    std::string get_name() const override { return "HtsWriter"; }
    stats::NamedStats sample_stats() const override;

    static OutputMode get_output_mode(const std::string& mode);

private:
    size_t total{0};
    size_t primary{0};
    size_t unmapped{0};
    size_t secondary{0};
    size_t supplementary{0};
    sam_hdr_t* m_header{nullptr};

    htsFile* m_file{nullptr};
    std::unique_ptr<std::thread> m_worker;
    void worker_thread();
    int write_hdr_sq(char* name, uint32_t length);
    int write(bam1_t* record);
    size_t m_num_reads_expected;
    std::unordered_set<std::string> m_processed_read_ids;
};

}  // namespace dorado
