#include "hts_writer/hts_file_writer.h"

#include "hts_utils/hts_file.h"
#include "hts_utils/hts_types.h"
#include "hts_writer/interface.h"
#include "utils/tty_utils.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>

namespace dorado {

namespace hts_writer {

namespace fs = std::filesystem;

HtsFileWriterBuilder::HtsFileWriterBuilder()
        : m_is_fd_tty(utils::is_fd_tty(stdout)), m_is_fd_pipe(utils::is_fd_pipe(stdout)) {}

HtsFileWriterBuilder::HtsFileWriterBuilder(bool emit_fastq,
                                           bool emit_sam,
                                           bool reference_requested,
                                           const std::optional<std::string> &output_dir,
                                           const int threads,
                                           const utils::ProgressCallback &progress_callback,
                                           const utils::DescriptionCallback &description_callback)
        : m_emit_fastq(emit_fastq),
          m_emit_sam(emit_sam),
          m_reference_requested(reference_requested),
          m_output_dir(output_dir),
          m_writer_threads(threads),
          m_progress_callback(progress_callback),
          m_description_callback(description_callback),
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
    } else {
        // Default case is to write BAMs - we can sort this if we have a reference and are using files.
        m_output_mode = OutputMode::BAM;
        m_sort = to_file && m_reference_requested && !m_is_fd_pipe;
    }
}

std::unique_ptr<HtsFileWriter> HtsFileWriterBuilder::build() {
    update();
    if (!m_output_dir.has_value()) {
        spdlog::warn("Creating StreamHtsFileWriter {}", to_string(m_output_mode));
        return std::make_unique<StreamHtsFileWriter>(m_output_mode, m_writer_threads,
                                                     m_progress_callback, m_description_callback);
    }
    spdlog::warn("Creating StructuredHtsFileWriter");
    return std::make_unique<StructuredHtsFileWriter>(m_output_dir.value(), m_output_mode,
                                                     m_writer_threads, m_sort, m_progress_callback,
                                                     m_description_callback);
}

void HtsFileWriter::process(const Processable item) {
    // Type-specific dispatch to handle(T)
    dispatch_processable(item, [this](const auto &t) { this->handle(t); });
};

StreamHtsFileWriter::StreamHtsFileWriter(OutputMode mode,
                                         int threads,
                                         const utils::ProgressCallback &progress_callback,
                                         const utils::DescriptionCallback &description_callback)
        : HtsFileWriter(mode, threads, progress_callback, description_callback) {};

void StreamHtsFileWriter::init() {
    // TODO: Use local m_mode;
};

void StreamHtsFileWriter::shutdown() {
    set_description("Finalising HTS output file");
    m_hts_file->finalise([this](size_t progress) { set_progress(progress); });
};

void StreamHtsFileWriter::handle(const HtsData &data) {
    if (m_hts_file == nullptr) {
        const bool sort_bam = false;
        m_hts_file = std::make_unique<utils::HtsFile>("-", utils::HtsFile::OutputMode::BAM,
                                                      m_threads, sort_bam);
        if (m_hts_file == nullptr) {
            std::runtime_error("Failed to create HTS output file");
        }

        if (m_header == nullptr) {
            std::logic_error("HtsFileWriter header not set before writing records.");
        }
        m_hts_file->set_header(m_header.get());
    }

    m_hts_file->write(data.bam_ptr.get());
};

/*

*/

StructuredHtsFileWriter::StructuredHtsFileWriter(
        const std::string &output_dir,
        OutputMode mode,
        int threads,
        bool sort,
        const utils::ProgressCallback &progress_callback,
        const utils::DescriptionCallback &description_callback)
        : HtsFileWriter(mode, threads, progress_callback, description_callback),
          m_output_dir(output_dir),
          m_sort(sort) {};

void StructuredHtsFileWriter::init() { try_create_output_folder(); };

void StructuredHtsFileWriter::shutdown() {
    set_description("Finalising HTS output files");

    size_t i = 0;
    const size_t n_files = m_hts_files.size();
    for (auto &[_, hts_file] : m_hts_files) {
        if (m_sort) {
            set_description("Sorting BAM");
        }

        const size_t index = i++;
        hts_file.finalise([this, index, n_files](size_t progress) {
            const size_t past_progress = index * size_t(100);
            const size_t p = std::min(size_t(100), (past_progress + progress) / n_files);
            set_progress(p);
        });
    }
};

void StructuredHtsFileWriter::handle(const HtsData &_) {
    if (m_header == nullptr) {
        std::logic_error("HtsFileWriter header not set before writing records.");
    }

    std::logic_error("StructuredHtsFileWriter::handle not implemented");
};

bool StructuredHtsFileWriter::try_create_output_folder() const {
    std::error_code creation_error;
    // N.B. No error code if folder already exists.
    fs::create_directories(fs::path{m_output_dir}, creation_error);
    if (creation_error) {
        spdlog::error("Unable to create output folder '{}'. ErrorCode({}) {}", m_output_dir,
                      creation_error.value(), creation_error.message());
        return false;
    }
    return true;
}

}  // namespace hts_writer

std::string to_string(hts_writer::OutputMode mode) {
    {
        switch (mode) {
        case hts_writer::OutputMode::BAM:
            return "BAM";
        case hts_writer::OutputMode::SAM:
            return "SAM";
        case hts_writer::OutputMode::FASTQ:
            return "FASTQ";
        }
    }
}

}  // namespace dorado
