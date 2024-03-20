#include "BarcodeDemuxerNode.h"

#include "read_pipeline/ReadPipeline.h"
#include "utils/SampleSheet.h"
#include "utils/hts_file.h"

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
        : MessageSink(10000, 1),
          m_output_dir(output_dir),
          m_htslib_threads(int(htslib_threads)),
          m_write_fastq(write_fastq),
          m_sample_sheet(std::move(sample_sheet)) {
    std::filesystem::create_directories(m_output_dir);
    start_input_processing(&BarcodeDemuxerNode::input_thread_fn, this);
}

BarcodeDemuxerNode::~BarcodeDemuxerNode() { stop_input_processing(); }

void BarcodeDemuxerNode::input_thread_fn() {
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
    auto& file = m_files[bc];
    if (!file) {
        // For new barcodes, create a new HTS file (either fastq or BAM).
        std::string filename = bc + (m_write_fastq ? ".fastq" : ".bam");
        auto filepath = m_output_dir / filename;
        auto filepath_str = filepath.string();

        file = std::make_unique<utils::HtsFile>(
                filepath_str,
                m_write_fastq ? utils::HtsFile::OutputMode::FASTQ : utils::HtsFile::OutputMode::BAM,
                m_htslib_threads);
        file->set_and_write_header(m_header.get());
    }

    auto hts_res = file->write(record);
    if (hts_res < 0) {
        throw std::runtime_error("Failed to write SAM record, error code " +
                                 std::to_string(hts_res));
    }

    m_processed_reads++;
    return hts_res;
}

void BarcodeDemuxerNode::set_header(const sam_hdr_t* const header) {
    if (header) {
        m_header.reset(sam_hdr_dup(header));
    }
}

void BarcodeDemuxerNode::finalise_hts_files(
        const utils::HtsFile::ProgressCallback& progress_callback) {
    const size_t num_files = m_files.size();
    size_t current_file_idx = 0;
    for (auto& [bc, hts_file] : m_files) {
        hts_file->finalise([&](size_t progress) {
            // Give each file/barcode the same contribution to the total progress.
            const size_t total_progress = (current_file_idx * 100 + progress) / num_files;
            progress_callback(total_progress);
        });
        ++current_file_idx;
    }

    m_files.clear();
    progress_callback(100);
}

stats::NamedStats BarcodeDemuxerNode::sample_stats() const {
    auto stats = stats::from_obj(m_work_queue);
    stats["demuxed_reads_written"] = m_processed_reads.load();
    return stats;
}

void BarcodeDemuxerNode::terminate(const FlushOptions&) { stop_input_processing(); }

}  // namespace dorado
