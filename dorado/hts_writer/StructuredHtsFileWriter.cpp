#include "hts_writer/StructuredHtsFileWriter.h"

#include "hts_utils/bam_utils.h"
#include "hts_utils/hts_file.h"
#include "hts_utils/hts_types.h"
#include "hts_writer/HtsFileWriter.h"
#include "hts_writer/Structure.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>

namespace dorado {

namespace hts_writer {

StructuredHtsFileWriter::StructuredHtsFileWriter(const HtsFileWriterConfig &cfg,
                                                 std::unique_ptr<IStructure> structure,
                                                 bool sort)
        : HtsFileWriter(cfg), m_structure(std::move(structure)), m_sort(sort) {};

void StructuredHtsFileWriter::shutdown() {
    if (std::exchange(m_has_shutdown, true)) {
        return;
    }

    set_description("Finalising outputs");
    size_t i = 0;
    const size_t n_files = m_hts_files.size();
    for (auto &[path, hts_file] : m_hts_files) {
        if (hts_file == nullptr) {
            spdlog::debug(
                    "StructuredHtsFileWriter::shutdown called on uninitialised hts_file - nothing "
                    "to do for '{}'",
                    path);
            continue;
        }
        const size_t index = i++;
        hts_file->finalise([this, index, n_files](size_t progress) {
            const size_t past_progress = index * size_t(100);
            const size_t p = std::min(size_t(100), (past_progress + progress) / n_files);
            set_progress(p);
        });
    }
}

bool StructuredHtsFileWriter::finalise_is_noop() const {
    return m_mode == OutputMode::FASTQ || !m_sort;
};

void StructuredHtsFileWriter::handle(const HtsData &item) {
    if (m_has_shutdown) {
        spdlog::debug("HtsFileWriter has shutdown and cannot process more work.");
        return;
    }

    const std::string path = m_structure->get_path(item);
    auto &hts_file = m_hts_files[path];
    if (!hts_file) {
        hts_file = std::make_unique<utils::HtsFile>(path, m_mode, m_threads, m_sort);
        set_hts_file_header(item, *hts_file);
    }

    hts_file->write(item.bam_ptr.get());
}

void StructuredHtsFileWriter::set_hts_file_header(const HtsData &item,
                                                  utils::HtsFile &hts_file) const {
    if (m_mode == OutputMode::FASTQ || m_mode == OutputMode::FASTA) {
        return;
    }

    if (m_mode == OutputMode::CRAM) {
        hts_file.set_cram_reference(m_reference);
    }

    if (m_shared_header != nullptr) {
        hts_file.set_header(m_shared_header.get());
        return;
    }

    if (m_dynamic_header != nullptr) {
        const auto &it = m_dynamic_header->find(item.read_attrs);
        if (it == m_dynamic_header->cend()) {
            spdlog::error("Failed to find dynamic header: RG='{}', runid='{}'",
                          utils::get_read_group_tag(item.bam_ptr.get()),
                          item.read_attrs.protocol_run_id);
            throw std::runtime_error("StructuredHtsFileWriter - Failed to load dynamic header.");
        }
        hts_file.set_header(it->second->get_merged_header());
        return;
    }

    throw std::logic_error("StructuredHtsFileWriter - header not set before writing records.");
}

}  // namespace hts_writer
}  // namespace dorado
