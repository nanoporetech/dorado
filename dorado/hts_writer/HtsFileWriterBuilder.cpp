#include "hts_writer/HtsFileWriterBuilder.h"

#include "hts_utils/hts_file.h"
#include "hts_writer/HtsFileWriter.h"
#include "hts_writer/StreamHtsFileWriter.h"
#include "hts_writer/Structure.h"
#include "hts_writer/StructuredHtsFileWriter.h"
#include "utils/SampleSheet.h"
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

HtsFileWriterBuilder::HtsFileWriterBuilder(bool emit_fastq,
                                           bool emit_sam,
                                           bool sort_requested,
                                           const std::optional<std::string>& output_dir,
                                           int threads,
                                           utils::ProgressCallback progress_callback,
                                           utils::DescriptionCallback description_callback,
                                           std::string gpu_names,
                                           std::shared_ptr<const utils::SampleSheet> sample_sheet)
        : m_emit_fastq(emit_fastq),
          m_emit_sam(emit_sam),
          m_sort_requested(sort_requested),
          m_output_dir(output_dir),
          m_writer_threads(threads),
          m_progress_callback(std::move(progress_callback)),
          m_description_callback(std::move(description_callback)),
          m_gpu_names(gpu_names.empty() ? "" : ("gpu:" + std::move(gpu_names))),
          m_sample_sheet(std::move(sample_sheet)),
          m_is_fd_tty(utils::is_fd_tty(stdout)),
          m_is_fd_pipe(utils::is_fd_pipe(stdout)) {
    if (m_emit_fastq && m_emit_sam) {
        spdlog::error("Only one of --emit-{fastq, sam} can be set (or none).");
        throw std::runtime_error("Invalid writer configuration");
    }
};

void HtsFileWriterBuilder::update() {
    const bool to_file = m_output_dir.has_value();

    if (m_emit_fastq) {
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
        m_sort = to_file && m_sort_requested;
    } else if (!to_file && m_is_fd_pipe) {
        // Write uncompressed zlib-wrapped BAM files when piping and emit-sam is not set
        m_output_mode = OutputMode::UBAM;
        m_sort = false;
    } else {
        // Default case is to write BAMs - we can sort this if we have a reference and are using files.
        m_output_mode = OutputMode::BAM;
        m_sort = to_file && m_sort_requested;
    }
}

std::unique_ptr<HtsFileWriter> HtsFileWriterBuilder::build() {
    update();
    const HtsFileWriterConfig cfg{m_output_mode, m_writer_threads, m_progress_callback,
                                  m_description_callback, m_gpu_names};

    if (!m_output_dir.has_value()) {
        return std::make_unique<StreamHtsFileWriter>(cfg);
    }

    auto nested_structure = std::make_unique<NestedFileStructure>(m_output_dir.value(),
                                                                  m_output_mode, m_sample_sheet);

    return std::make_unique<StructuredHtsFileWriter>(cfg, std::move(nested_structure), m_sort);
}

OutputMode HtsFileWriterBuilder::get_output_mode() {
    update();
    return m_output_mode;
}

BasecallHtsFileWriterBuilder::BasecallHtsFileWriterBuilder(
        bool emit_fastq,
        bool emit_sam,
        bool reference_requested,
        const std::optional<std::string>& output_dir,
        int writer_threads,
        utils::ProgressCallback progress_callback,
        utils::DescriptionCallback description_callback,
        std::string gpu_names,
        std::shared_ptr<const utils::SampleSheet> sample_sheet)
        : HtsFileWriterBuilder(emit_fastq,
                               emit_sam,
                               reference_requested,
                               output_dir,
                               writer_threads,
                               std::move(progress_callback),
                               std::move(description_callback),
                               std::move(gpu_names),
                               std::move(sample_sheet)) {
    if (emit_fastq && reference_requested) {
        spdlog::error(
                "--emit-fastq cannot be used with --reference as FASTQ cannot store "
                "alignment results.");
        throw std::runtime_error("Invalid writer configuration");
    }
};

DemuxHtsFileWriterBuilder::DemuxHtsFileWriterBuilder(
        bool emit_fastq,
        bool sort_requested,
        const std::optional<std::string>& output_dir,
        int writer_threads,
        utils::ProgressCallback progress_callback,
        utils::DescriptionCallback description_callback,
        std::string gpu_names,
        std::shared_ptr<const utils::SampleSheet> sample_sheet)
        : HtsFileWriterBuilder(emit_fastq,
                               false,
                               sort_requested,
                               output_dir,
                               writer_threads,
                               std::move(progress_callback),
                               std::move(description_callback),
                               std::move(gpu_names),
                               std::move(sample_sheet)) {};

AlignerHtsFileWriterBuilder::AlignerHtsFileWriterBuilder(
        bool emit_sam,
        bool sort_requested,
        const std::optional<std::string>& output_dir,
        int writer_threads,
        utils::ProgressCallback progress_callback,
        utils::DescriptionCallback description_callback)
        : HtsFileWriterBuilder(false,
                               emit_sam,
                               sort_requested,
                               output_dir,
                               writer_threads,
                               std::move(progress_callback),
                               std::move(description_callback),
                               std::string(),
                               nullptr) {};

}  // namespace hts_writer
}  // namespace dorado
