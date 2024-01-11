#include "HtsWriter.h"

#include "read_pipeline/ReadPipeline.h"
#include "utils/sequence_utils.h"

#include <htslib/bgzf.h>
#include <htslib/kroundup.h>
#include <htslib/sam.h>
#include <indicators/progress_bar.hpp>
#include <spdlog/spdlog.h>

#include <stdexcept>
#include <string>
#include <unordered_set>

namespace dorado {

HtsWriter::HtsWriter(const std::string& filename, OutputMode mode, size_t threads)
        : MessageSink(10000) {
    switch (mode) {
    case OutputMode::FASTQ:
        m_file = hts_open(filename.c_str(), "wf");
        break;
    case OutputMode::BAM:
        m_file = hts_open(filename.c_str(), "wb");
        break;
    case OutputMode::SAM:
        m_file = hts_open(filename.c_str(), "w");
        break;
    case OutputMode::UBAM:
        m_file = hts_open(filename.c_str(), "wb0");
        break;
    default:
        throw std::runtime_error("Unknown output mode selected: " +
                                 std::to_string(static_cast<int>(mode)));
    }
    if (!m_file) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    if (m_file->format.compression == bgzf) {
        auto res = bgzf_mt(m_file->fp.bgzf, int(threads), 128);
        if (res < 0) {
            throw std::runtime_error("Could not enable multi threading for BAM generation.");
        }
    }
    start_threads();
}

void HtsWriter::start_threads() {
    m_worker = std::make_unique<std::thread>(std::thread(&HtsWriter::worker_thread, this));
}

void HtsWriter::terminate_impl() {
    terminate_input_queue();
    if (m_worker && m_worker->joinable()) {
        m_worker->join();
    }
    m_worker.reset();
}

void HtsWriter::restart() {
    restart_input_queue();
    start_threads();
}

HtsWriter::~HtsWriter() {
    terminate_impl();
    sam_hdr_destroy(m_header);
    hts_close(m_file);
}

HtsWriter::OutputMode HtsWriter::get_output_mode(const std::string& mode) {
    if (mode == "sam") {
        return OutputMode::SAM;
    } else if (mode == "bam") {
        return OutputMode::BAM;
    } else if (mode == "fastq") {
        return OutputMode::FASTQ;
    }
    throw std::runtime_error("Unknown output mode: " + mode);
}

void HtsWriter::worker_thread() {
    Message message;
    while (get_input_message(message)) {
        // If this message isn't a BamPtr, ignore it.
        if (!std::holds_alternative<BamPtr>(message)) {
            continue;
        }

        auto aln = std::move(std::get<BamPtr>(message));
        write(aln.get());

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

            m_processed_read_ids.insert(std::move(read_id));
        }
    }
}

int HtsWriter::write(bam1_t* const record) {
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

    // FIXME -- HtsWriter is constructed in a state where attempting to write
    // will segfault, since set_and_write_header has to have been called
    // in order to set m_header.
    assert(m_header);
    auto res = sam_write1(m_file, m_header, record);
    if (res < 0) {
        throw std::runtime_error("Failed to write SAM record, error code " + std::to_string(res));
    }
    return res;
}

int HtsWriter::set_and_write_header(const sam_hdr_t* const header) {
    if (header) {
        // Avoid leaking memory if this is called twice.
        if (m_header) {
            sam_hdr_destroy(m_header);
        }
        m_header = sam_hdr_dup(header);
        return sam_hdr_write(m_file, m_header);
    }
    return 0;
}

stats::NamedStats HtsWriter::sample_stats() const {
    auto stats = stats::from_obj(m_work_queue);
    stats["unique_simplex_reads_written"] = double(m_processed_read_ids.size());
    stats["duplex_reads_written"] = m_duplex_reads_written.load();
    stats["split_reads_written"] = m_split_reads_written.load();
    return stats;
}

}  // namespace dorado
