#include "HtsWriter.h"

#include "htslib/bgzf.h"
#include "htslib/kroundup.h"
#include "htslib/sam.h"
#include "read_pipeline/ReadPipeline.h"
#include "utils/sequence_utils.h"

#include <indicators/progress_bar.hpp>
#include <spdlog/spdlog.h>

#include <stdexcept>
#include <string>
#include <unordered_set>

namespace dorado {

HtsWriter::HtsWriter(const std::string& filename, OutputMode mode, size_t threads, size_t num_reads, const sam_hdr_t* header)
        : MessageSink(10000), m_num_reads_expected(num_reads), m_header(sam_hdr_dup(header)) {
    switch (mode) {
    case FASTQ:
        m_file = hts_open(filename.c_str(), "wf");
        break;
    case BAM:
        m_file = hts_open(filename.c_str(), "wb");
        break;
    case SAM:
        m_file = hts_open(filename.c_str(), "w");
        break;
    case UBAM:
        m_file = hts_open(filename.c_str(), "wb0");
        break;
    default:
        throw std::runtime_error("Unknown output mode selected: " + std::to_string(mode));
    }
    if (!m_file) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    if (m_file->format.compression == bgzf) {
        auto res = bgzf_mt(m_file->fp.bgzf, threads, 128);
        if (res < 0) {
            throw std::runtime_error("Could not enable multi threading for BAM generation.");
        }
    }

    if (sam_hdr_write(m_file, m_header)) {
        throw std::runtime_error("Failed to write header");
    }

    m_worker = std::make_unique<std::thread>(std::thread(&HtsWriter::worker_thread, this));
}

HtsWriter::~HtsWriter() {
    // Adding for thread safety in case worker thread throws exception.
    terminate();
    if (m_worker->joinable()) {
        m_worker->join();
    }
    sam_hdr_destroy(m_header);
    hts_close(m_file);
    spdlog::info("> total/primary/unmapped {}/{}/{}", total, primary, unmapped);
}

HtsWriter::OutputMode HtsWriter::get_output_mode(const std::string& mode) {
    if (mode == "sam") {
        return SAM;
    } else if (mode == "bam") {
        return BAM;
    } else if (mode == "fastq") {
        return FASTQ;
    }
    throw std::runtime_error("Unknown output mode: " + mode);
}

void HtsWriter::worker_thread() {
    size_t write_count = 0;

    Message message;
    while (m_work_queue.try_pop(message)) {
        auto aln = std::get<BamPtr>(std::move(message));
        write(aln.get());
        std::string read_id = bam_get_qname(aln.get());
        aln.reset();  // Free the bam alignment that's already written

        // For the purpose of estimating write count, we ignore duplex reads
        // these can be identified by a semicolon in their ID.
        // TODO: This is a hack, we should have a better way of identifying duplex reads.
        bool ignore_read_id = read_id.find(';') != std::string::npos;

        if (!ignore_read_id) {
            m_processed_read_ids.insert(std::move(read_id));
        }
    }
    spdlog::debug("Written {} records.", write_count);
}

int HtsWriter::write(bam1_t* record) {
    // track stats
    total++;
    if (record->core.flag & BAM_FUNMAP) {
        unmapped++;
    }
    if (record->core.flag & BAM_FSECONDARY) {
        secondary++;
    }
    if (record->core.flag & BAM_FSUPPLEMENTARY) {
        supplementary++;
    }
    primary = total - secondary - supplementary - unmapped;

    auto res = sam_write1(m_file, m_header, record);
    if (res < 0) {
        throw std::runtime_error("Failed to write SAM record, error code " + std::to_string(res));
    }
    return res;
}

stats::NamedStats HtsWriter::sample_stats() const {
    auto stats = stats::from_obj(m_work_queue);
    stats["unique_simplex_reads_written"] = m_processed_read_ids.size();
    return stats;
}

}  // namespace dorado
