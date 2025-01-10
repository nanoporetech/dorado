#include "BarcodeDemuxerNode.h"

#include "read_pipeline/ReadPipeline.h"
#include "utils/SampleSheet.h"
#include "utils/fastq_reader.h"
#include "utils/hts_file.h"

#include <htslib/bgzf.h>
#include <htslib/sam.h>

#include <cassert>
#include <stdexcept>
#include <string>

namespace dorado {

namespace {
constexpr size_t BAM_BUFFER_SIZE =
        20000000;  // 20 MB per barcode classification. So roughly 2 GB for 96 barcodes.

std::string get_run_id_from_fq_tag(const bam1_t& record) {
    auto fastq_id_tag = bam_aux_get(&record, "fq");
    if (!fastq_id_tag) {
        return {};
    }
    const std::string header{bam_aux2Z(fastq_id_tag)};
    utils::FastqRecord fastq_record{};
    if (!fastq_record.set_header(header)) {
        return {};
    }
    return std::string{fastq_record.run_id_view()};
}

std::string get_run_id_from_rg_tag(const bam1_t& record) {
    const auto read_group_tag = bam_aux_get(&record, "RG");
    if (read_group_tag) {
        const std::string read_group_string = std::string(bam_aux2Z(read_group_tag));
        auto pos = read_group_string.find('_');
        if (pos != std::string::npos) {
            return read_group_string.substr(0, pos);
        }
    }

    return {};
}

std::string get_run_id(const bam1_t& record) {
    auto run_id = get_run_id_from_rg_tag(record);
    if (run_id.empty()) {
        run_id = get_run_id_from_fq_tag(record);
    }
    return run_id.empty() ? "unknown_run_id" : run_id;
}

void apply_sample_sheet_alias(const utils::SampleSheet& sample_sheet,
                              std::string& barcode,
                              bam1_t& record) {
    // experiment id and position id are not stored in the bam record, so we can't recover them to use here
    const auto alias = sample_sheet.get_alias("", "", "", barcode);
    if (!alias.empty()) {
        barcode = alias;
        bam_aux_update_str(&record, "BC", int(barcode.size() + 1), barcode.c_str());
    }
}

}  // namespace

BarcodeDemuxerNode::BarcodeDemuxerNode(const std::string& output_dir,
                                       size_t htslib_threads,
                                       bool write_fastq,
                                       std::unique_ptr<const utils::SampleSheet> sample_sheet,
                                       bool sort_bam)
        : MessageSink(10000, 1),
          m_output_dir(output_dir),
          m_htslib_threads(int(htslib_threads)),
          m_write_fastq(write_fastq),
          m_sort_bam(sort_bam && !write_fastq),
          m_sample_sheet(std::move(sample_sheet)) {
    std::filesystem::create_directories(m_output_dir);
}

BarcodeDemuxerNode::~BarcodeDemuxerNode() { stop_input_processing(); }

void BarcodeDemuxerNode::input_thread_fn() {
    Message message;
    while (get_input_message(message)) {
        auto bam_message = std::move(std::get<BamMessage>(message));
        write(*bam_message.bam_ptr);
    }
}

// Each barcode is mapped to its own file. Depending
// on the barcode assigned to each read, the read is
// written to the corresponding barcode file.
int BarcodeDemuxerNode::write(bam1_t& record) {
    assert(m_header);
    // Fetch the barcode name.
    std::string barcode = UNCLASSIFIED;
    auto bam_tag = bam_aux_get(&record, "BC");
    if (bam_tag) {
        barcode = std::string(bam_aux2Z(bam_tag));
    }
    if (m_sample_sheet) {
        apply_sample_sheet_alias(*m_sample_sheet, barcode, record);
    }

    const auto run_id = get_run_id(record);

    // Check for existence of file for that barcode and run id.
    auto& file = m_files[run_id + barcode];
    if (!file) {
        // For new barcodes, create a new HTS file (either fastq or BAM).
        const std::string filename = run_id + "_" + barcode + (m_write_fastq ? ".fastq" : ".bam");
        const auto filepath = m_output_dir / filename;
        const auto filepath_str = filepath.string();

        file = std::make_unique<utils::HtsFile>(
                filepath_str,
                m_write_fastq ? utils::HtsFile::OutputMode::FASTQ : utils::HtsFile::OutputMode::BAM,
                m_htslib_threads, m_sort_bam);
        if (m_sort_bam) {
            file->set_buffer_size(BAM_BUFFER_SIZE);
        }
        file->set_header(m_header.get());
    }

    auto hts_res = file->write(&record);
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
