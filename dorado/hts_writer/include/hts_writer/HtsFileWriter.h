#pragma once

#include "hts_utils/hts_file.h"
#include "interface.h"
#include "utils/stats.h"

#include <string>
#include <utility>

namespace dorado {

class HtsData;

namespace hts_writer {

namespace ext {
constexpr std::string_view BAM{".bam"};
constexpr std::string_view SAM{".sam"};
constexpr std::string_view FASTQ{".fastq"};
}  // namespace ext

using OutputMode = utils::HtsFile::OutputMode;

struct HtsFileWriterConfig {
    OutputMode mode;
    int threads;
    utils::ProgressCallback progress_callback;
    utils::DescriptionCallback description_callback;
    std::string gpu_names;
};

class HtsFileWriter : public IWriter {
public:
    explicit HtsFileWriter(const HtsFileWriterConfig& cfg)
            : m_mode(cfg.mode),
              m_threads(cfg.threads),
              m_progress_callback(cfg.progress_callback),
              m_description_callback(cfg.description_callback),
              m_gpu_names(cfg.gpu_names) {}
    std::string get_name() const override { return "HtsFileWriter"; }
    stats::NamedStats sample_stats() const override;

    void set_header(SamHdrSharedPtr header) { m_header = std::move(header); };
    SamHdrSharedPtr& get_header() { return m_header; }

    void process(const Processable item) override;

    OutputMode get_mode() const { return m_mode; }
    virtual bool finalise_is_noop() const = 0;

    int get_threads() const { return m_threads; }
    const std::string& get_gpu_names() const { return m_gpu_names; }

    void set_progress(size_t progress) const { m_progress_callback(progress); }
    void set_description(const std::string& description) const {
        m_description_callback(description);
    }

protected:
    const OutputMode m_mode;
    const int m_threads;
    const utils::ProgressCallback m_progress_callback;
    const utils::DescriptionCallback m_description_callback;

    SamHdrSharedPtr m_header{nullptr};

    std::string m_gpu_names{};

    void prepare_item(const HtsData& item) const;
    virtual void handle(const HtsData& item) = 0;
    void update_stats(const HtsData& item);

    // Stats counters
    std::atomic<std::size_t> m_primary_simplex_reads_written{0};
    std::atomic<std::size_t> m_duplex_reads_written{0};
    std::atomic<std::size_t> m_split_reads_written{0};
};

}  // namespace hts_writer

}  // namespace dorado
