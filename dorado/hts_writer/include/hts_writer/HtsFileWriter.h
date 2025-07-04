#pragma once

#include "hts_utils/hts_file.h"
#include "hts_utils/hts_types.h"
#include "interface.h"

#include <string>

namespace dorado {

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

    void take_header(SamHdrPtr header) { m_header = std::move(header); };
    SamHdrPtr& header() { return m_header; }
    void process(const Processable item);

    OutputMode mode() const { return m_mode; }
    virtual bool finalise_is_noop() const = 0;

    int get_threads() const { return m_threads; }

    void set_progress(size_t progress) const { m_progress_callback(progress); }
    void set_description(const std::string& description) const {
        m_description_callback(description);
    }

protected:
    const OutputMode m_mode;
    const int m_threads;
    const utils::ProgressCallback m_progress_callback;
    const utils::DescriptionCallback m_description_callback;
    SamHdrPtr m_header{nullptr};

    virtual void handle(const HtsData& _) = 0;
};

}  // namespace hts_writer

}  // namespace dorado
