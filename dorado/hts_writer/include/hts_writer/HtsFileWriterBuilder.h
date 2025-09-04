#pragma once

#include "HtsFileWriter.h"

#include <memory>

namespace dorado {

namespace utils {
class SampleSheet;
}

namespace hts_writer {

class HtsFileWriter;

class HtsFileWriterBuilder {
protected:
    HtsFileWriterBuilder(bool emit_fastq,
                         bool emit_sam,
                         bool sort_requested,
                         const std::optional<std::string>& output_dir,
                         int writer_threads,
                         utils::ProgressCallback progress_callback,
                         utils::DescriptionCallback description_callback,
                         std::string gpu_names,
                         std::shared_ptr<const utils::SampleSheet> sample_sheet);

public:
    OutputMode get_output_mode();

    void update();
    std::unique_ptr<HtsFileWriter> build();

private:
    const bool m_emit_fastq{false}, m_emit_sam{false}, m_sort_requested{false};
    const std::optional<std::string> m_output_dir{std::nullopt};
    const int m_writer_threads{0};
    const utils::ProgressCallback m_progress_callback;
    const utils::DescriptionCallback m_description_callback;
    const std::string m_gpu_names;
    const std::shared_ptr<const utils::SampleSheet> m_sample_sheet;

    bool m_sort{false};
    OutputMode m_output_mode{OutputMode::BAM};

protected:
    bool m_is_fd_tty{false}, m_is_fd_pipe{false};
};

class BasecallHtsFileWriterBuilder final : public HtsFileWriterBuilder {
public:
    BasecallHtsFileWriterBuilder(bool emit_fastq,
                                 bool emit_sam,
                                 bool reference_requested,
                                 const std::optional<std::string>& output_dir,
                                 int writer_threads,
                                 utils::ProgressCallback progress_callback,
                                 utils::DescriptionCallback description_callback,
                                 std::string gpu_names,
                                 std::shared_ptr<const utils::SampleSheet> sample_sheet);
};

class DemuxHtsFileWriterBuilder final : public HtsFileWriterBuilder {
public:
    DemuxHtsFileWriterBuilder(bool emit_fastq,
                              bool sort_requested,
                              const std::optional<std::string>& output_dir,
                              int writer_threads,
                              utils::ProgressCallback progress_callback,
                              utils::DescriptionCallback description_callback,
                              std::string gpu_names,
                              std::shared_ptr<const utils::SampleSheet> sample_sheet);
};

}  // namespace hts_writer
}  // namespace dorado