#include "HtsWriter.h"

#include "read_pipeline/ReadPipeline.h"
#include "utils/sequence_utils.h"

#include <htslib/bgzf.h>
#include <htslib/kroundup.h>
#include <htslib/sam.h>
#include <indicators/progress_bar.hpp>
#include <spdlog/spdlog.h>

#include <cassert>
#include <filesystem>
#include <stdexcept>

namespace dorado {

using OutputMode = dorado::utils::HtsFile::OutputMode;

HtsWriter::HtsWriter(const std::string& filename, OutputMode mode, size_t threads)
        : MessageSink(10000, 1), m_file(std::make_unique<utils::HtsFile>(filename, mode, threads)) {
    start_input_processing(&HtsWriter::input_thread_fn, this);
}

HtsWriter::~HtsWriter() { stop_input_processing(); }

OutputMode HtsWriter::get_output_mode(const std::string& mode) {
    if (mode == "sam") {
        return OutputMode::SAM;
    } else if (mode == "bam") {
        return OutputMode::BAM;
    } else if (mode == "fastq") {
        return OutputMode::FASTQ;
    }
    throw std::runtime_error("Unknown output mode: " + mode);
}

void HtsWriter::input_thread_fn() {
    Message message;
    while (get_input_message(message)) {
        // If this message isn't a BamPtr, ignore it.
        if (!std::holds_alternative<BamPtr>(message)) {
            continue;
        }

        auto aln = std::move(std::get<BamPtr>(message));
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

        bool ignore_read_id = dx_tag == 1;

        if (ignore_read_id) {
            // Read is a duplex read.
            m_duplex_reads_written++;
        } else {
            std::string read_id;

            // If read is a split read, use the parent read id
            // to track write count since we don't know a priori
            // how many split reads will be generated.
            auto pid_tag = bam_aux_get(aln.get(), "pi");
            if (pid_tag) {
                read_id = std::string(bam_aux2Z(pid_tag));
                m_split_reads_written++;
            } else {
                read_id = bam_get_qname(aln.get());
            }

            m_processed_read_ids.add(std::move(read_id));
        }
    }
}

int HtsWriter::write(const bam1_t* const record) {
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

    return m_file->write(record);
}

int HtsWriter::set_and_write_header(const sam_hdr_t* const header) {
    return m_file->set_and_write_header(header);
}

stats::NamedStats HtsWriter::sample_stats() const {
    auto stats = stats::from_obj(m_work_queue);
    stats["unique_simplex_reads_written"] = static_cast<double>(m_processed_read_ids.size());
    stats["duplex_reads_written"] = static_cast<double>(m_duplex_reads_written.load());
    stats["split_reads_written"] = static_cast<double>(m_split_reads_written.load());
    return stats;
}

void HtsWriter::terminate(const FlushOptions&) {
    stop_input_processing();
    m_file.reset();
}

std::size_t HtsWriter::ProcessedReadIds::size() const { return m_threadsafe_count_of_reads; }

void HtsWriter::ProcessedReadIds::add(std::string read_id) {
    read_ids.insert(std::move(read_id));
    m_threadsafe_count_of_reads = read_ids.size();
}

}  // namespace dorado
