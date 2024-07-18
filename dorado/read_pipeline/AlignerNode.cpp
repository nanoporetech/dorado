#include "AlignerNode.h"

#include "ClientInfo.h"
#include "alignment/Minimap2Aligner.h"
#include "alignment/Minimap2Index.h"
#include "alignment/alignment_info.h"
#include "alignment/minimap2_args.h"
#include "messages.h"

#include <htslib/sam.h>
#include <minimap.h>
#include <spdlog/spdlog.h>

#include <cassert>
#include <filesystem>
#include <string>
#include <vector>

namespace {

std::shared_ptr<const dorado::alignment::Minimap2Index> load_and_get_index(
        dorado::alignment::IndexFileAccess& index_file_access,
        const std::string& index_file,
        const dorado::alignment::Minimap2Options& options,
        const int threads) {
    int num_index_construction_threads{
            dorado::alignment::mm2::print_aln_seq() ? 1 : static_cast<int>(threads)};
    switch (index_file_access.load_index(index_file, options, num_index_construction_threads)) {
    case dorado::alignment::IndexLoadResult::reference_file_not_found:
        throw std::runtime_error("AlignerNode reference path does not exist: " + index_file);
    case dorado::alignment::IndexLoadResult::validation_error:
        throw std::runtime_error("AlignerNode validation error checking minimap options");
    case dorado::alignment::IndexLoadResult::split_index_not_supported:
        throw std::runtime_error(
                "Dorado doesn't support split index for alignment. Please re-run with larger index "
                "size.");
    case dorado::alignment::IndexLoadResult::no_index_loaded:
    case dorado::alignment::IndexLoadResult::end_of_index:
        throw std::runtime_error("AlignerNode index loading error - should not reach here.");
    case dorado::alignment::IndexLoadResult::success:
        break;
    }
    return index_file_access.get_index(index_file, options);
}

void update_bed_results(dorado::ReadCommon& read_common, const dorado::alignment::BedFile& bed) {
    for (auto& align_result : read_common.alignment_results) {
        for (const auto& entry : bed.entries(align_result.genome)) {
            if (!(entry.start > (size_t)align_result.genome_end ||
                  entry.end < (size_t)align_result.genome_start) &&
                (entry.strand == align_result.direction || entry.strand == '.')) {
                // A hit
                align_result.bed_hits++;
                if (!align_result.bed_lines.empty()) {
                    align_result.bed_lines += "\n";
                }
                align_result.bed_lines += entry.bed_line;
            }
        }
    }
}

}  // namespace

namespace dorado {

AlignerNode::AlignerNode(std::shared_ptr<alignment::IndexFileAccess> index_file_access,
                         std::shared_ptr<alignment::BedFileAccess> bed_file_access,
                         const std::string& index_file,
                         const std::string& bed_file,
                         const alignment::Minimap2Options& options,
                         int threads)
        : MessageSink(10000, threads),
          m_index_for_bam_messages(
                  load_and_get_index(*index_file_access, index_file, options, threads)),
          m_index_file_access(std::move(index_file_access)),
          m_bed_file_access(std::move(bed_file_access)) {
    if (!bed_file.empty()) {
        if (!m_bed_file_access) {
            throw std::runtime_error(
                    "Bed-file has been specified, but no bed-file loader "
                    "has been provided.");
        }
        m_bedfile_for_bam_messages = m_bed_file_access->get_bedfile(bed_file);
        if (!m_bedfile_for_bam_messages) {
            throw std::runtime_error("Expected bed-file " + bed_file + " is not loaded.");
        }
        auto header_sequence_records = m_index_for_bam_messages->get_sequence_records_for_header();
        for (const auto& entry : header_sequence_records) {
            m_header_sequence_names.emplace_back(entry.first);
        }
    }
}

AlignerNode::AlignerNode(std::shared_ptr<alignment::IndexFileAccess> index_file_access,
                         std::shared_ptr<alignment::BedFileAccess> bed_file_access,
                         int threads)
        : MessageSink(10000, threads),
          m_index_file_access(std::move(index_file_access)),
          m_bed_file_access(std::move(bed_file_access)) {}

std::shared_ptr<const alignment::Minimap2Index> AlignerNode::get_index(
        const ClientInfo& client_info) {
    auto align_info = client_info.contexts().get_ptr<const alignment::AlignmentInfo>();
    if (!align_info || align_info->reference_file.empty()) {
        return {};
    }
    auto index =
            m_index_file_access->get_index(align_info->reference_file, align_info->minimap_options);
    if (!index) {
        if (client_info.is_disconnected()) {
            // Unlikely but ... may have disconnected since last checked and caused a
            // an unload of the index file.
            return {};
        }
        throw std::runtime_error(
                "Cannot align read. Expected alignment reference file is not loaded: " +
                align_info->reference_file);
    }

    return index;
}

std::shared_ptr<alignment::BedFile> AlignerNode::get_bedfile(const ClientInfo& client_info,
                                                             const std::string& bedfile) {
    if (m_bed_file_access && !bedfile.empty()) {
        return m_bed_file_access->get_bedfile(bedfile);
    }
    if (bedfile.empty() || client_info.is_disconnected()) {
        // Unlikely but ... may have disconnected since last checked and caused a
        // an unload of the index file.
        return {};
    }
    throw std::runtime_error("Expected bed-file is not loaded: " + bedfile);
}

alignment::HeaderSequenceRecords AlignerNode::get_sequence_records_for_header() const {
    assert(m_index_for_bam_messages != nullptr &&
           "get_sequence_records_for_header only valid if AlignerNode constructed with index file");
    return alignment::Minimap2Aligner(m_index_for_bam_messages).get_sequence_records_for_header();
}

void AlignerNode::align_read_common(ReadCommon& read_common, mm_tbuf_t* tbuf) {
    // Note: This code path is only used by the basecall server.
    if (read_common.client_info->is_disconnected()) {
        return;
    }

    auto align_info = read_common.client_info->contexts().get_ptr<const alignment::AlignmentInfo>();
    if (!align_info) {
        return;
    }

    auto index = get_index(*read_common.client_info);
    if (!index) {
        return;
    }

    alignment::Minimap2Aligner(index).align(read_common, align_info->alignment_header, tbuf);

    auto bed = get_bedfile(*read_common.client_info, align_info->bed_file);
    if (bed) {
        update_bed_results(read_common, *bed);
    }
}

void AlignerNode::input_thread_fn() {
    Message message;
    mm_tbuf_t* tbuf = mm_tbuf_init();
    auto align_read = [this, tbuf](auto&& read) {
        align_read_common(read->read_common, tbuf);
        send_message_to_sink(std::move(read));
    };
    while (get_input_message(message)) {
        if (std::holds_alternative<BamMessage>(message)) {
            auto bam_message = std::get<BamMessage>(std::move(message));
            auto records = alignment::Minimap2Aligner(m_index_for_bam_messages)
                                   .align(bam_message.bam_ptr.get(), tbuf);
            for (auto& record : records) {
                if (m_bedfile_for_bam_messages && !(record->core.flag & BAM_FUNMAP)) {
                    auto ref_id = record->core.tid;
                    add_bed_hits_to_record(m_header_sequence_names.at(ref_id), record.get());
                }
                send_message_to_sink(BamMessage{std::move(record), bam_message.client_info});
            }
        } else if (std::holds_alternative<SimplexReadPtr>(message)) {
            align_read(std::get<SimplexReadPtr>(std::move(message)));
        } else if (std::holds_alternative<DuplexReadPtr>(message)) {
            align_read(std::get<DuplexReadPtr>(std::move(message)));
        } else {
            send_message_to_sink(std::move(message));
            continue;
        }
    }
    mm_tbuf_destroy(tbuf);
}

stats::NamedStats AlignerNode::sample_stats() const { return stats::from_obj(m_work_queue); }

void AlignerNode::add_bed_hits_to_record(const std::string& genome, bam1_t* record) {
    size_t genome_start = record->core.pos;
    size_t genome_end = bam_endpos(record);
    char direction = (bam_is_rev(record)) ? '-' : '+';
    int bed_hits = 0;
    for (const auto& interval : m_bedfile_for_bam_messages->entries(genome)) {
        if (!(interval.start >= genome_end || interval.end <= genome_start) &&
            (interval.strand == direction || interval.strand == '.')) {
            bed_hits++;
        }
    }
    // update the record.
    bam_aux_append(record, "bh", 'i', sizeof(bed_hits), (uint8_t*)&bed_hits);
}

}  // namespace dorado
