#pragma once

#include "HtsFileWriter.h"

namespace dorado {
namespace hts_writer {

class HtsFileWriter;

class HtsFileWriterBuilder {
public:
    HtsFileWriterBuilder(bool emit_fastq,
                         bool emit_sam,
                         bool reference_requested,
                         const std::optional<std::string>& output_dir,
                         int writer_threads,
                         utils::ProgressCallback progress_callback,
                         utils::DescriptionCallback description_callback,
                         std::string gpu_names);

    void set_emit_fastq(bool emit_fastq);
    bool get_emit_fastq() const;

    void set_emit_sam(bool emit_sam);
    bool get_emit_sam() const;

    void set_reference_requested(bool reference_requested);
    bool get_reference_requested() const;

    void set_output_dir(const std::optional<std::string>& output_dir);
    std::optional<std::string> get_output_dir() const;

    void set_writer_threads(int threads);
    int get_writer_threads() const;

    void set_progress_callback(const utils::ProgressCallback& callback);
    void set_description_callback(const utils::DescriptionCallback& callback);

    void set_gpu_names(bool gpu_names);
    const std::string& get_gpu_names() const;

    void set_is_fd_tty(bool is_fd_tty);
    void set_is_fd_pipe(bool is_fd_pipe);

    bool get_sort();
    OutputMode get_output_mode();

    void update();
    std::unique_ptr<HtsFileWriter> build();

private:
    bool m_emit_fastq{false}, m_emit_sam{false}, m_reference_requested{false};
    std::optional<std::string> m_output_dir{std::nullopt};
    int m_writer_threads{0};
    utils::ProgressCallback m_progress_callback;
    utils::DescriptionCallback m_description_callback;
    std::string m_gpu_names;
    bool m_is_fd_tty{false}, m_is_fd_pipe{false};

    bool m_sort{false};
    OutputMode m_output_mode{OutputMode::BAM};
};

}  // namespace hts_writer
}  // namespace dorado