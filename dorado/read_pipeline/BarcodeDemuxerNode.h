#pragma once

#include "MessageSink.h"
#include "utils/hts_file.h"
#include "utils/stats.h"
#include "utils/types.h"

#include <atomic>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>

struct bam1_t;

namespace dorado {

namespace utils {
class SampleSheet;
}

class BarcodeDemuxerNode : public MessageSink {
public:
    using HtsFiles = std::unordered_map<std::string, std::unique_ptr<utils::HtsFile>>;

    BarcodeDemuxerNode(const std::string& output_dir,
                       size_t htslib_threads,
                       bool write_fastq,
                       std::unique_ptr<const utils::SampleSheet> sample_sheet,
                       bool sort_bam);
    ~BarcodeDemuxerNode();
    std::string get_name() const override { return "BarcodeDemuxerNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions&) override;
    void restart() override {
        start_input_processing([this] { input_thread_fn(); }, "brcd_demux");
    }

    void set_header(const sam_hdr_t* header);

    // Finalisation must occur before destruction of this node.
    // Note that this isn't safe to call until after this node has been terminated.
    void finalise_hts_files(const utils::HtsFile::ProgressCallback& progress_callback);

private:
    const std::filesystem::path m_output_dir;
    const int m_htslib_threads;
    SamHdrPtr m_header;
    std::atomic<int> m_processed_reads{0};

    HtsFiles m_files;
    void input_thread_fn();
    int write(bam1_t& record);
    const bool m_write_fastq;
    const bool m_sort_bam;
    const std::unique_ptr<const utils::SampleSheet> m_sample_sheet;
};

}  // namespace dorado
