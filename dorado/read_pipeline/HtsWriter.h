#pragma once
#include "htslib/sam.h"
#include "read_pipeline/ReadPipeline.h"
#include "utils/stats.h"

#ifdef WIN32
#include <indicators/progress_bar.hpp>
#else
#include <indicators/block_progress_bar.hpp>
#endif

#include <memory>
#include <string>

namespace dorado {

class HtsWriter : public MessageSink {
public:
    enum OutputMode {
        UBAM,
        BAM,
        SAM,
        FASTQ,
    };

    HtsWriter(const std::string& filename, OutputMode mode, size_t threads, size_t num_reads);
    ~HtsWriter();
    std::string get_name() const override { return "HtsWriter"; }
    stats::NamedStats sample_stats() const override;
    int write_header(const sam_hdr_t* header);
    int write(bam1_t* record);
    void join();

    static OutputMode get_output_mode(std::string mode);

    size_t total{0};
    size_t primary{0};
    size_t unmapped{0};
    size_t secondary{0};
    size_t supplementary{0};
    sam_hdr_t* header{nullptr};

private:
    htsFile* m_file{nullptr};
    std::unique_ptr<std::thread> m_worker;
    void worker_thread();
    int write_hdr_sq(char* name, uint32_t length);
    bool m_prog_bar_initialized{false};
    size_t m_num_reads_expected;
    int m_progress_bar_interval;

#ifdef WIN32
    indicators::ProgressBar m_progress_bar {
#else
    indicators::BlockProgressBar m_progress_bar{
#endif
        indicators::option::Stream{std::cerr}, indicators::option::BarWidth{30},
                indicators::option::ShowElapsedTime{true},
                indicators::option::ShowRemainingTime{true},
                indicators::option::ShowPercentage{true},
    };
};

}  // namespace dorado
