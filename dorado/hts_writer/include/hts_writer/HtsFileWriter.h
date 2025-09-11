#pragma once

#include "hts_utils/HeaderMapper.h"
#include "hts_utils/hts_file.h"
#include "interface.h"
#include "utils/stats.h"

#include <memory>
#include <string>

namespace dorado {

class HtsData;

namespace hts_writer {

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

    // Set a single header to write to all output files
    void set_shared_header(SamHdrSharedPtr header);
    // Set a lookup for pre-built output headers based indexed on read attributes at file write time.
    void set_dynamic_header(const std::shared_ptr<utils::HeaderMapper::HeaderMap>& header_map);

    void process(const Processable item) override;

    OutputMode get_mode() const { return m_mode; }
    virtual bool finalise_is_noop() const = 0;

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

    SamHdrSharedPtr m_shared_header{nullptr};
    std::shared_ptr<utils::HeaderMapper::HeaderMap> m_dynamic_header{nullptr};

    std::string m_gpu_names{};

    void prepare_item(const HtsData& item) const;
    virtual void handle(const HtsData& item) = 0;
    void update_stats(const HtsData& item);

    // Stats counters
    std::atomic<std::size_t> m_total_records_written{0};
    std::atomic<std::size_t> m_primary_records_written{0};
    std::atomic<std::size_t> m_unmapped_records_written{0};
    std::atomic<std::size_t> m_secondary_records_written{0};
    std::atomic<std::size_t> m_supplementary_records_written{0};

    std::atomic<std::size_t> m_primary_simplex_reads_written{0};
    std::atomic<std::size_t> m_duplex_reads_written{0};
    std::atomic<std::size_t> m_split_reads_written{0};
};

}  // namespace hts_writer

}  // namespace dorado
