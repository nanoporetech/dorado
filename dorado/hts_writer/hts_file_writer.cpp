#include "hts_writer/hts_file_writer.h"

#include "hts_utils/hts_file.h"
#include "hts_utils/hts_types.h"
#include "hts_writer/interface.h"
#include "utils/tty_utils.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>

namespace dorado {

namespace hts_writer {

namespace fs = std::filesystem;

void HtsFileWriter::process(const Processable item) {
    // Type-specific dispatch to handle(T)
    dispatch_processable(item, [this](const auto &t) { this->handle(t); });
};

HtsFileWriterBuilder::HtsFileWriterBuilder(bool emit_fastq,
                                           bool emit_sam,
                                           bool reference_requested,
                                           const std::optional<std::string> &output_dir,
                                           const int threads,
                                           utils::ProgressCallback &callback) {
    this->set_output_from_cli(emit_fastq, emit_sam, reference_requested, output_dir);
    this->set_writer_threads(threads);
    this->set_progress_callback(callback);
};

void HtsFileWriterBuilder::set_output_from_cli(bool emit_fastq,
                                               bool emit_sam,
                                               bool reference_requested,
                                               const std::optional<std::string> &output_dir) {
    this->set_output_dir(output_dir);

    if (emit_fastq) {
        if (emit_sam) {
            spdlog::error("Only one of --emit-{fastq, sam} can be set (or none).");
            std::runtime_error("Invalid writer configuration");
        }
        if (reference_requested) {
            spdlog::error(
                    "--emit-fastq cannot be used with --reference as FASTQ cannot store "
                    "alignment results.");
            std::runtime_error("Invalid writer configuration");
        }
        spdlog::info(" - Note: FASTQ output is not recommended as not all data can be preserved.");
        this->set_output_mode(OutputMode::FASTQ);
    } else if (emit_sam || (output_dir == std::nullopt && utils::is_fd_tty(stdout))) {
        this->set_output_mode(OutputMode::SAM);
    } else if (output_dir == std::nullopt && utils::is_fd_pipe(stdout)) {
        this->set_output_mode(OutputMode::BAM);
    } else {
        this->set_output_mode(OutputMode::BAM);
        this->set_sort_bam(true);
    }
}

std::unique_ptr<HtsFileWriter> HtsFileWriterBuilder::build() const {
    if (!m_output_dir.has_value()) {
        return std::make_unique<StreamHtsFileWriter>(m_output_mode, m_threads, m_callback);
    }
    return std::make_unique<StructuredHtsFileWriter>(m_output_dir.value(), m_output_mode, m_threads,
                                                     m_callback, m_sort_bam);
}

StreamHtsFileWriter::StreamHtsFileWriter(OutputMode mode,
                                         int threads,
                                         const utils::ProgressCallback &callback)
        : HtsFileWriter(mode, threads, callback) {};

void StreamHtsFileWriter::init() {
    // TODO: Use local m_mode;
    const bool sort_bam = false;
    m_hts_file = std::make_unique<utils::HtsFile>("-", utils::HtsFile::OutputMode::BAM, m_threads,
                                                  sort_bam);
};

void StreamHtsFileWriter::shutdown() {
    m_hts_file->finalise([this](size_t progress) { update_progress(progress); });
};

void StreamHtsFileWriter::handle(const HtsData &data) {
    if (m_header == nullptr) {
        std::logic_error("HtsFileWriter header not set before writing records.");
    }

    m_hts_file->write(data.bam_ptr.get());
};

/*

*/

StructuredHtsFileWriter::StructuredHtsFileWriter(const std::string &output_dir,
                                                 OutputMode mode,
                                                 int threads,
                                                 const utils::ProgressCallback &callback,
                                                 bool sort)
        : HtsFileWriter(mode, threads, callback), m_output_dir(output_dir), m_sort(sort) {};

void StructuredHtsFileWriter::init() { try_create_output_folder(); };

void StructuredHtsFileWriter::shutdown() {
    size_t i = 0;
    const size_t n_files = m_hts_files.size();
    for (auto &[_, hts_file] : m_hts_files) {
        const size_t index = i++;
        hts_file.finalise([this, index, n_files](size_t progress) {
            const size_t past_progress = index * size_t(100);
            const size_t p = std::min(size_t(100), (past_progress + progress) / n_files);
            update_progress(p);
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
}  // namespace dorado
