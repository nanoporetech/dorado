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

    OutputMode get_output_mode() const { return m_output_mode; }

    void set_output_dir(const std::optional<std::string>& output_dir) { m_output_dir = output_dir; }
    std::optional<std::string> get_output_dir() const { return m_output_dir; }

    bool get_sort() const { return m_sort; }

    void set_writer_threads(int threads) { m_writer_threads = threads; }
    int get_writer_threads() const { return m_writer_threads; }

    void set_progress_callback(const utils::ProgressCallback& callback) {
        m_progress_callback = callback;
    }
    void set_description_callback(const utils::DescriptionCallback& callback) {
        m_description_callback = callback;
    }

    void set_is_fd_tty(bool is_fd_tty) { m_is_fd_tty = is_fd_tty; }
    void set_is_fd_pipe(bool is_fd_pipe) { m_is_fd_pipe = is_fd_pipe; }

    void update();
    std::unique_ptr<HtsFileWriter> build();

private:
    bool m_emit_fastq{false}, m_emit_sam{false}, m_reference_requested{false};
    std::optional<std::string> m_output_dir{std::nullopt};
    int m_writer_threads{0};
    utils::ProgressCallback m_progress_callback;
    utils::DescriptionCallback m_description_callback;

    std::string m_gpu_names;

    bool m_sort{false};

    OutputMode m_output_mode{OutputMode::BAM};
    bool m_is_fd_tty{false}, m_is_fd_pipe{false};
};

}  // namespace hts_writer
}  // namespace dorado