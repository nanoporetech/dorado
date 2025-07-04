#include "hts_writer/HtsFileWriter.h"

#include "hts_utils/bam_utils.h"
#include "hts_utils/hts_types.h"

#include <htslib/sam.h>

namespace dorado {
namespace hts_writer {

void HtsFileWriter::process(const Processable item) {
    // Type-specific dispatch to handle(T)
    dispatch_processable(item, [this](const auto &t) { this->prepare_item(t); });
    dispatch_processable(item, [this](const auto &t) { this->handle(t); });
    dispatch_processable(item, [this](const auto &t) { this->tally(t); });
}

void HtsFileWriter::prepare_item(const HtsData &item) const {
    if (m_mode == OutputMode::FASTQ && !m_gpu_names.empty()) {
        bam_aux_append(item.bam_ptr.get(), "DS", 'Z', int(m_gpu_names.length() + 1),
                       (uint8_t *)m_gpu_names.c_str());
    }

    // Verify that the MN tag, if it exists, and the sequence length are in sync.
    if (auto tag = bam_aux_get(item.bam_ptr.get(), "MN"); tag != nullptr) {
        if (bam_aux2i(tag) != item.bam_ptr.get()->core.l_qseq) {
            throw std::runtime_error("MN tag and sequence length are not in sync.");
        };
    }
}

void HtsFileWriter::tally(const HtsData &data) {
    const auto &aln = data.bam_ptr.get();

    const uint16_t flags = aln->core.flag;
    const bool is_unmapped = ((flags & BAM_FUNMAP) != 0);
    const bool is_primary = ((flags & BAM_FSECONDARY) == 0) && ((flags & BAM_FSUPPLEMENTARY) == 0);

    if (utils::is_duplex_record(data.bam_ptr.get())) {
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
        if ((data.subread_id == 0) && (is_unmapped || is_primary)) {
            m_primary_simplex_reads_written.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

stats::NamedStats HtsFileWriter::sample_stats() const {
    stats::NamedStats stats;
    stats["unique_simplex_reads_written"] =
            static_cast<double>(m_primary_simplex_reads_written.load());
    stats["duplex_reads_written"] = static_cast<double>(m_duplex_reads_written.load());
    stats["split_reads_written"] = static_cast<double>(m_split_reads_written.load());
    return stats;
}

}  // namespace hts_writer
}  // namespace dorado
