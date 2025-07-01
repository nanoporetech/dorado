#include "hts_writer/HtsFileWriterBuilder.h"

#include "hts_utils/hts_file.h"
#include "hts_writer/hts_file_writer.h"
#include "utils/tty_utils.h"

#include <spdlog/spdlog.h>

#include <cstdio>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace dorado {

namespace hts_writer {

namespace fs = std::filesystem;

HtsFileWriterBuilder::HtsFileWriterBuilder()
        : m_is_fd_tty(utils::is_fd_tty(stdout)), m_is_fd_pipe(utils::is_fd_pipe(stdout)) {}

HtsFileWriterBuilder::HtsFileWriterBuilder(bool emit_fastq,
                                           bool emit_sam,
                                           bool reference_requested,
                                           const std::optional<std::string> &output_dir,
                                           int threads,
                                           utils::ProgressCallback progress_callback,
                                           utils::DescriptionCallback description_callback)
        : m_emit_fastq(emit_fastq),
          m_emit_sam(emit_sam),
          m_reference_requested(reference_requested),
          m_output_dir(output_dir),
          m_writer_threads(threads),
          m_progress_callback(std::move(progress_callback)),
          m_description_callback(std::move(description_callback)),
          m_is_fd_tty(utils::is_fd_tty(stdout)),
          m_is_fd_pipe(utils::is_fd_pipe(stdout)) {};

void HtsFileWriterBuilder::update() {
    const bool to_file = m_output_dir.has_value();

    if (m_emit_fastq) {
        if (m_emit_sam) {
            spdlog::error("Only one of --emit-{fastq, sam} can be set (or none).");
            std::runtime_error("Invalid writer configuration");
        }
        if (m_reference_requested) {
            spdlog::error(
                    "--emit-fastq cannot be used with --reference as FASTQ cannot store "
                    "alignment results.");
            std::runtime_error("Invalid writer configuration");
        }
        spdlog::info(" - Note: FASTQ output is not recommended as not all data can be preserved.");
        m_output_mode = OutputMode::FASTQ;
        m_sort = false;
    } else if (!to_file && m_is_fd_tty) {
        // Write SAM files when writing to a tty - this cannot be sorted
        m_output_mode = OutputMode::SAM;
        m_sort = false;
    } else if (m_emit_sam) {
        // Write SAM if chosen by user - sort only if we have a reference and are writing to a file
        m_output_mode = OutputMode::SAM;
        m_sort = to_file && m_reference_requested;
    } else if (!to_file && m_is_fd_pipe) {
        // Write uncompressed zlib-wrapped BAM files when piping and emit-sam is not set
        m_output_mode = OutputMode::UBAM;
        m_sort = false;
    } else {
        // Default case is to write BAMs - we can sort this if we have a reference and are using files.
        m_output_mode = OutputMode::BAM;
        m_sort = to_file && m_reference_requested;
    }
}

std::unique_ptr<HtsFileWriter> HtsFileWriterBuilder::build() {
    update();
    if (!m_output_dir.has_value()) {
        spdlog::info("Creating StreamHtsFileWriter {}", to_string(m_output_mode));
        return std::make_unique<SingleHtsFileWriter>(m_output_mode, m_writer_threads,
                                                     m_progress_callback, m_description_callback);
    }
    spdlog::error("Creating StructuredHtsFileWriter");
    return std::make_unique<StructuredHtsFileWriter>(m_output_dir.value(), m_output_mode,
                                                     m_writer_threads, m_sort, m_progress_callback,
                                                     m_description_callback);
}
}  // namespace hts_writer
}  // namespace dorado
