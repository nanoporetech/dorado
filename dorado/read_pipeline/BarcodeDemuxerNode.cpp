#include "BarcodeDemuxerNode.h"

#include "read_pipeline/ReadPipeline.h"
#include "utils/SampleSheet.h"

#include <htslib/bgzf.h>
#include <htslib/sam.h>

#include <cassert>
#include <stdexcept>
#include <string>

namespace dorado {

BarcodeDemuxerNode::BarcodeDemuxerNode(const std::string& output_dir,
                                       size_t htslib_threads,
                                       bool write_fastq,
                                       std::unique_ptr<const utils::SampleSheet> sample_sheet)
        : MessageSink(10000),
          m_output_dir(output_dir),
          m_htslib_threads(int(htslib_threads)),
          m_write_fastq(write_fastq),
          m_sample_sheet(std::move(sample_sheet)) {
    std::filesystem::create_directories(m_output_dir);
    start_threads();
}

void BarcodeDemuxerNode::start_threads() {
    m_worker = std::make_unique<std::thread>(std::thread(&BarcodeDemuxerNode::worker_thread, this));
}

void BarcodeDemuxerNode::terminate_impl() {
    terminate_input_queue();
    if (m_worker && m_worker->joinable()) {
        m_worker->join();
    }
}

void BarcodeDemuxerNode::restart() {
    restart_input_queue();
    start_threads();
}

BarcodeDemuxerNode::~BarcodeDemuxerNode() {
    terminate_impl();
    sam_hdr_destroy(m_header);
    for (auto& [k, f] : m_files) {
        hts_close(f);
    }
}

void BarcodeDemuxerNode::worker_thread() {
    Message message;
    while (get_input_message(message)) {
        auto aln = std::move(std::get<BamPtr>(message));
        write(aln.get());
    }
}

// Each barcode is mapped to its own file. Depending
// on the barcode assigned to each read, the read is
// written to the corresponding barcode file.
int BarcodeDemuxerNode::write(bam1_t* const record) {
    assert(m_header);
    // Fetch the barcode name.
    std::string bc = "unclassified";
    auto bam_tag = bam_aux_get(record, "BC");
    if (bam_tag) {
        bc = std::string(bam_aux2Z(bam_tag));
    }

    if (m_sample_sheet) {
        // experiment id and position id are not stored in the bam record, so we can't recover them to use here
        auto alias = m_sample_sheet->get_alias("", "", "", bc);
        if (!alias.empty()) {
            bc = alias;
            bam_aux_update_str(record, "BC", int(bc.size() + 1), bc.c_str());
        }
    }
    // Check of existence of file for that barcode.
    auto res = m_files.find(bc);
    htsFile* file = nullptr;
    if (res != m_files.end()) {
        file = res->second;
    } else {
        // For new barcodes, create a new HTS file (either fastq or BAM).
        std::string filename = bc + (m_write_fastq ? ".fastq" : ".bam");
        auto filepath = m_output_dir / filename;
        auto filepath_str = filepath.string();
        file = hts_open(filepath_str.c_str(), (m_write_fastq ? "wf" : "wb"));
        if (!file) {
            throw std::runtime_error("Failed to open new HTS output file at " + filepath.string());
        }
        if (file->format.compression == bgzf) {
            auto bgz_res = bgzf_mt(file->fp.bgzf, m_htslib_threads, 128);
            if (bgz_res < 0) {
                throw std::runtime_error("Could not enable multi threading for BAM generation.");
            }
        }
        m_files[bc] = file;
        auto hts_res = sam_hdr_write(file, m_header);
        if (hts_res < 0) {
            throw std::runtime_error("Failed to write SAM header, error code " +
                                     std::to_string(hts_res));
        }
    }
    auto hts_res = sam_write1(file, m_header, record);
    if (hts_res < 0) {
        throw std::runtime_error("Failed to write SAM record, error code " +
                                 std::to_string(hts_res));
    }
    m_processed_reads++;
    return hts_res;
}

void BarcodeDemuxerNode::set_header(const sam_hdr_t* const header) {
    if (header) {
        // Avoid leaking memory if this is called twice.
        if (m_header) {
            sam_hdr_destroy(m_header);
        }
        m_header = sam_hdr_dup(header);
    }
}

stats::NamedStats BarcodeDemuxerNode::sample_stats() const {
    auto stats = stats::from_obj(m_work_queue);
    stats["demuxed_reads_written"] = m_processed_reads.load();
    return stats;
}

}  // namespace dorado
