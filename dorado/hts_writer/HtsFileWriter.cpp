#include "hts_writer/HtsFileWriter.h"

#include "hts_utils/bam_utils.h"
#include "hts_utils/hts_types.h"
#include "utils/time_utils.h"

#include <htslib/sam.h>

#include <atomic>

namespace dorado {
namespace hts_writer {

void HtsFileWriter::process(const Processable item) {
    // Type-specific dispatch to handle(T)
    dispatch_processable(item, [this](const auto &t) { this->prepare_item(t); });
    dispatch_processable(item, [this](const auto &t) { this->handle(t); });
    dispatch_processable(item, [this](const auto &t) { this->update_stats(t); });
}

void HtsFileWriter::prepare_item(const HtsData &hts_data) const {
    if (m_mode == OutputMode::FASTQ) {
        if (!m_gpu_names.empty()) {
            bam_aux_append(hts_data.bam_ptr.get(), "DS", 'Z', int(m_gpu_names.length() + 1),
                           reinterpret_cast<const uint8_t *>(m_gpu_names.c_str()));
        }
        if (!hts_data.read_attrs.flowcell_id.empty()) {
            bam_aux_append(
                    hts_data.bam_ptr.get(), "PU", 'Z',
                    int(hts_data.read_attrs.flowcell_id.length() + 1),
                    reinterpret_cast<const uint8_t *>(hts_data.read_attrs.flowcell_id.c_str()));
        }
        std::string exp_start_time_str = utils::get_string_timestamp_from_unix_time(
                hts_data.read_attrs.protocol_start_time_ms);
        bam_aux_append(hts_data.bam_ptr.get(), "DT", 'Z', int(exp_start_time_str.length() + 1),
                       reinterpret_cast<const uint8_t *>(exp_start_time_str.c_str()));
    }

    // Verify that the MN tag, if it exists, and the sequence length are in sync.
    if (auto tag = bam_aux_get(hts_data.bam_ptr.get(), "MN"); tag != nullptr) {
        if (bam_aux2i(tag) != hts_data.bam_ptr.get()->core.l_qseq) {
            throw std::runtime_error("MN tag and sequence length are not in sync.");
        };
    }
}

void HtsFileWriter::update_stats(const HtsData &hts_data) {
    const auto &aln = hts_data.bam_ptr.get();

    const uint16_t flags = aln->core.flag;
    const bool is_unmapped = ((flags & BAM_FUNMAP) != 0);
    const bool is_primary = ((flags & BAM_FSECONDARY) == 0) && ((flags & BAM_FSUPPLEMENTARY) == 0);

    if (utils::is_duplex_record(hts_data.bam_ptr.get())) {
        // For the purpose of estimating write count, we ignore duplex reads
        m_duplex_reads_written.fetch_add(1, std::memory_order_relaxed);
    } else {
        // We can end up with multiple records for a single input read because
        // of read splitting or alignment.
        // If read is a split read, only count it if the subread id is 0
        // If read is an aligned read, only count it if it's the primary alignment
        const auto pid_tag = bam_aux_get(aln, "pi");
        if (pid_tag) {
            m_split_reads_written.fetch_add(1, std::memory_order_relaxed);
        }
        if ((hts_data.read_attrs.subread_id == 0) && (is_unmapped || is_primary)) {
            m_primary_simplex_reads_written.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

stats::NamedStats HtsFileWriter::sample_stats() const {
    stats::NamedStats stats;
    stats["unique_simplex_reads_written"] =
            static_cast<double>(m_primary_simplex_reads_written.load(std::memory_order_relaxed));
    stats["duplex_reads_written"] =
            static_cast<double>(m_duplex_reads_written.load(std::memory_order_relaxed));
    stats["split_reads_written"] =
            static_cast<double>(m_split_reads_written.load(std::memory_order_relaxed));
    return stats;
}

}  // namespace hts_writer
}  // namespace dorado
