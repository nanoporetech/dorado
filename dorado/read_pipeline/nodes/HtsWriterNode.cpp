#include "read_pipeline/nodes/HtsWriterNode.h"

#include <htslib/bgzf.h>
#include <htslib/kroundup.h>
#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <cassert>
#include <stdexcept>

namespace dorado {

using OutputMode = dorado::utils::HtsFile::OutputMode;

HtsWriterNode::HtsWriterNode(utils::HtsFile& file, std::string gpu_names)
        : MessageSink(10000, 1), m_file(file), m_gpu_names(std::move(gpu_names)) {
    if (!m_gpu_names.empty()) {
        m_gpu_names = "gpu:" + m_gpu_names;
    }
}

HtsWriterNode::~HtsWriterNode() { stop_input_processing(utils::AsyncQueueTerminateFast::Yes); }

OutputMode HtsWriterNode::get_output_mode(const std::string& mode) {
    if (mode == "sam") {
        return OutputMode::SAM;
    } else if (mode == "bam") {
        return OutputMode::BAM;
    } else if (mode == "fastq") {
        return OutputMode::FASTQ;
    }
    throw std::runtime_error("Unknown output mode: " + mode);
}

void HtsWriterNode::input_thread_fn() {
    Message message;
    while (get_input_message(message)) {
        if (!std::holds_alternative<BamMessage>(message)) {
            continue;
        }

        auto bam_message = std::move(std::get<BamMessage>(message));
        BamPtr aln = std::move(bam_message.data.bam_ptr);

        if (m_file.get_output_mode() == utils::HtsFile::OutputMode::FASTQ) {
            if (!m_gpu_names.empty()) {
                bam_aux_append(aln.get(), "DS", 'Z', int(m_gpu_names.length() + 1),
                               (uint8_t*)m_gpu_names.c_str());
            }
        }

        auto res = write(aln.get());
        if (res < 0) {
            throw std::runtime_error("Failed to write SAM record, error code " +
                                     std::to_string(res));
        }

        // For the purpose of estimating write count, we ignore duplex reads
        int64_t dx_tag = 0;
        auto tag_str = bam_aux_get(aln.get(), "dx");
        if (tag_str) {
            dx_tag = bam_aux2i(tag_str);
        }

        const uint16_t flags = aln->core.flag;

        const bool is_unmapped = ((flags & BAM_FUNMAP) != 0);
        const bool is_primary =
                ((flags & BAM_FSECONDARY) == 0) && ((flags & BAM_FSUPPLEMENTARY) == 0);
        const bool is_duplex = (dx_tag == 1);

        if (is_duplex) {
            // Read is a duplex read.
            m_duplex_reads_written.fetch_add(1, std::memory_order_relaxed);
        } else {
            // We can end up with multiple records for a single input read because
            // of read splitting or alignment.
            // If read is a split read, only count it if the subread id is 0
            // If read is an aligned read, only count it if it's the primary alignment
            const auto pid_tag = bam_aux_get(aln.get(), "pi");
            if (pid_tag) {
                m_split_reads_written.fetch_add(1, std::memory_order_relaxed);
            }
            if ((bam_message.data.subread_id == 0) && (is_unmapped || is_primary)) {
                m_primary_simplex_reads_written.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
}

int HtsWriterNode::write(bam1_t* const record) {
    // track stats
    m_total++;
    if (record->core.flag & BAM_FUNMAP) {
        m_unmapped++;
    }
    if (record->core.flag & BAM_FSECONDARY) {
        m_secondary++;
    }
    if (record->core.flag & BAM_FSUPPLEMENTARY) {
        m_supplementary++;
    }
    m_primary = m_total - m_secondary - m_supplementary - m_unmapped;

    // Verify that the MN tag, if it exists, and the sequence length are in sync.
    if (auto tag = bam_aux_get(record, "MN"); tag != nullptr) {
        if (bam_aux2i(tag) != record->core.l_qseq) {
            throw std::runtime_error("MN tag and sequence length are not in sync.");
        };
    }

    return m_file.write(record);
}

std::string HtsWriterNode::get_name() const { return "HtsWriterNode"; }

stats::NamedStats HtsWriterNode::sample_stats() const {
    stats::NamedStats stats = MessageSink::sample_stats();
    stats["unique_simplex_reads_written"] =
            static_cast<double>(m_primary_simplex_reads_written.load());
    stats["duplex_reads_written"] = static_cast<double>(m_duplex_reads_written.load());
    stats["split_reads_written"] = static_cast<double>(m_split_reads_written.load());
    return stats;
}

void HtsWriterNode::terminate(const TerminateOptions& terminate_options) {
    stop_input_processing(terminate_options.fast);
}

void HtsWriterNode::restart() {
    start_input_processing([this] { input_thread_fn(); }, "hts_writer");
}

}  // namespace dorado
