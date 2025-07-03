#include "hts_writer/StructuredHtsFileWriter.h"

#include "hts_utils/hts_file.h"
#include "hts_writer/StructureStrategy.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

namespace dorado {

namespace hts_writer {

namespace fs = std::filesystem;

StructuredHtsFileWriter::StructuredHtsFileWriter(OutputMode mode,
                                                 std::unique_ptr<IStructure> structure,
                                                 int threads,
                                                 bool sort,
                                                 utils::ProgressCallback progress_callback,
                                                 utils::DescriptionCallback description_callback)
        : HtsFileWriter(mode,
                        threads,
                        std::move(progress_callback),
                        std::move(description_callback)),
          m_structure(std::move(structure)),
          m_sort(sort) {};

void StructuredHtsFileWriter::init() { m_structure->init(); }

void StructuredHtsFileWriter::shutdown() {
    set_description("Finalising outputs");
    size_t i = 0;
    const size_t n_files = m_hts_files.size();
    for (auto &[_, hts_file] : m_hts_files) {
        const size_t index = i++;
        hts_file->finalise([this, index, n_files](size_t progress) {
            const size_t past_progress = index * size_t(100);
            const size_t p = std::min(size_t(100), (past_progress + progress) / n_files);
            set_progress(p);
        });
    }
}

void StructuredHtsFileWriter::handle(const HtsData &item) {
    if (m_header == nullptr) {
        std::logic_error("HtsFileWriter header not set before writing records.");
    }

    // Implemented only for SingleFileStructure while structured outputs are under development
    auto &structure_ref = *m_structure;
    if (typeid(structure_ref) != typeid(SingleFileStructure)) {
        std::logic_error("StructuredHtsFileWriter is only implemented for SingleFileStructure");
    }

    const std::shared_ptr<const fs::path> path = m_structure->get_path(item);
    if (!m_hts_files.contains(*path)) {
        auto hts_file = std::make_unique<utils::HtsFile>(path->string(), m_mode, m_threads, m_sort);
        if (hts_file == nullptr) {
            std::runtime_error("Failed to create HTS output file at: '" + path->string() + "'.");
        }
        hts_file->set_header(m_header.get());
        m_hts_files.emplace(*path, std::move(hts_file));
    }

    std::unique_ptr<utils::HtsFile> &hts_file = m_hts_files.at(*path);
    hts_file->write(item.bam_ptr.get());
}

}  // namespace hts_writer
}  // namespace dorado
