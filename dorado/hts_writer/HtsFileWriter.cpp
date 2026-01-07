#include "hts_writer/HtsFileWriter.h"

#include "hts_utils/bam_utils.h"
#include "hts_utils/hts_types.h"
#include "utils/barcode_kits.h"
#include "utils/time_utils.h"

#include <htslib/sam.h>

#include <atomic>
#include <stdexcept>

namespace dorado {
namespace hts_writer {

void HtsFileWriter::process(const Processable item) {
    // Type-specific dispatch to handle(T)
    dispatch_processable(item, [this](auto &t) { this->prepare_item(t); });
    dispatch_processable(item, [this](const auto &t) { this->handle(t); });
    dispatch_processable(item, [this](const auto &t) { this->update_stats(t); });
}

void HtsFileWriter::prepare_item(HtsData &hts_data) const {
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
        std::string exp_start_time_str = utils::get_string_timestamp_from_unix_time_ms(
                hts_data.read_attrs.protocol_start_time_ms);
        bam_aux_append(hts_data.bam_ptr.get(), "DT", 'Z', int(exp_start_time_str.length() + 1),
                       reinterpret_cast<const uint8_t *>(exp_start_time_str.c_str()));

        if (hts_data.barcoding_result &&
            hts_data.barcoding_result->barcode_name != UNCLASSIFIED_STR) {
            std::string barcode_name =
                    barcode_kits::normalize_barcode_name(hts_data.barcoding_result->barcode_name);
            std::string_view alias = hts_data.barcoding_result->alias.empty()
                                             ? barcode_name
                                             : hts_data.barcoding_result->alias;
            bam_aux_update_str(hts_data.bam_ptr.get(), "SM",
                               static_cast<int>(barcode_name.length() + 1), barcode_name.c_str());
            bam_aux_update_str(hts_data.bam_ptr.get(), "al", static_cast<int>(alias.length() + 1),
                               alias.data());
        }
    }

    // Verify that the MN tag, if it exists, and the sequence length are in sync.
    if (auto tag = bam_aux_get(hts_data.bam_ptr.get(), "MN"); tag != nullptr) {
        if (bam_aux2i(tag) != hts_data.bam_ptr.get()->core.l_qseq) {
            throw std::runtime_error("MN tag and sequence length are not in sync.");
        };
    }
}

namespace {
inline void atomic_increment(std::atomic<std::size_t> &v) {
    v.fetch_add(1, std::memory_order_relaxed);
}
}  // namespace

void HtsFileWriter::update_stats(const HtsData &hts_data) {
    const auto &aln = hts_data.bam_ptr.get();
    const uint16_t flags = aln->core.flag;
    const bool is_unmapped = ((flags & BAM_FUNMAP) != 0);
    const bool is_secondary = (flags & BAM_FSECONDARY) != 0;
    const bool is_supplementary = (flags & BAM_FSUPPLEMENTARY) != 0;
    const bool is_primary = !is_secondary && !is_supplementary;

    atomic_increment(m_total_records_written);
    if (is_unmapped) {
        atomic_increment(m_unmapped_records_written);
    }
    if (is_secondary) {
        atomic_increment(m_secondary_records_written);
    }
    if (is_supplementary) {
        atomic_increment(m_supplementary_records_written);
    }
    if (is_primary) {
        atomic_increment(m_primary_records_written);
    }

    if (utils::is_duplex_record(hts_data.bam_ptr.get())) {
        // For the purpose of estimating write count, we ignore duplex reads
        atomic_increment(m_duplex_reads_written);
        return;
    }
    // We can end up with multiple records for a single input read because
    // of read splitting or alignment.
    // If read is a split read, only count it if the subread id is 0
    // If read is an aligned read, only count it if it's the primary alignment
    const auto pid_tag = bam_aux_get(aln, "pi");
    if (pid_tag) {
        atomic_increment(m_split_reads_written);
    }
    if ((hts_data.read_attrs.subread_id == 0) && (is_unmapped || is_primary)) {
        atomic_increment(m_primary_simplex_reads_written);
    }
}

namespace {
inline double atomic_load(const std::atomic<std::size_t> &v) {
    return static_cast<double>(v.load(std::memory_order_relaxed));
}
}  // namespace

stats::NamedStats HtsFileWriter::sample_stats() const {
    stats::NamedStats stats;
    stats["total_records_written"] = atomic_load(m_total_records_written);
    stats["unmapped_records_written"] = atomic_load(m_unmapped_records_written);
    stats["secondary_records_written"] = atomic_load(m_secondary_records_written);
    stats["supplementary_records_written"] = atomic_load(m_supplementary_records_written);
    stats["primary_records_written"] = atomic_load(m_primary_records_written);

    stats["unique_simplex_reads_written"] = atomic_load(m_primary_simplex_reads_written);
    stats["duplex_reads_written"] = atomic_load(m_duplex_reads_written);
    stats["split_reads_written"] = atomic_load(m_split_reads_written);
    return stats;
}

void HtsFileWriter::set_shared_header(SamHdrSharedPtr header) {
    if (m_dynamic_header != nullptr) {
        throw std::logic_error("set_shared_header is incompatible with set_dynamic_header.");
    }
    m_shared_header = std::move(header);
};

void HtsFileWriter::set_dynamic_header(
        const std::shared_ptr<const utils::HeaderMapper::HeaderMap> &header_map) {
    if (m_shared_header != nullptr) {
        throw std::logic_error("set_dynamic_header is incompatible with set_shared_header.");
    }
    m_dynamic_header = header_map;
};

}  // namespace hts_writer
}  // namespace dorado
