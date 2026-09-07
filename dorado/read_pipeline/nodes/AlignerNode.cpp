#include "read_pipeline/nodes/AlignerNode.h"

#include "aligner/Minimap2Aligner.h"
#include "alignment/Minimap2Index.h"
#include "alignment/Minimap2MemoryLimiter.h"
#include "alignment/alignment_info.h"
#include "alignment/minimap2_args.h"
#include "read_pipeline/base/ClientInfo.h"
#include "read_pipeline/base/messages/DuplexRead.h"
#include "read_pipeline/base/messages/SimplexRead.h"
#include "utils/context_container.h"
#include "utils/sequence_utils.h"

#include <htslib/sam.h>
#include <minimap.h>

#include <cassert>
#include <string>
#include <vector>

namespace dorado {

namespace {

std::shared_ptr<const dorado::alignment::Minimap2Index> load_and_get_index(
        dorado::alignment::IndexFileAccess& index_file_access,
        const std::string& index_file,
        const dorado::alignment::Minimap2Options& options) {
    // This is a blocking operation on the main thread during the construction of the AlignerNode,
    // so use all the cores to get it loaded.
    const int threads = std::thread::hardware_concurrency();
    int num_index_construction_threads{
            dorado::alignment::mm2::print_aln_seq() ? 1 : static_cast<int>(threads)};
    switch (index_file_access.load_index(index_file, options, num_index_construction_threads)) {
    case dorado::alignment::IndexLoadResult::reference_file_not_found:
        throw std::runtime_error("AlignerNode reference path does not exist: " + index_file);
    case dorado::alignment::IndexLoadResult::validation_error:
        throw std::runtime_error("AlignerNode validation error checking minimap options");
    case dorado::alignment::IndexLoadResult::file_open_error:
        throw std::runtime_error("Error opening index file: " + index_file);
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
                    align_result.bed_lines += '\n';
                }
                align_result.bed_lines += entry.bed_line;
            }
        }
    }
}

}  // namespace

AlignerNode::AlignerNode(std::shared_ptr<alignment::IndexFileAccess> index_file_access,
                         std::shared_ptr<alignment::BedFileAccess> bed_file_access,
                         const std::string& index_file,
                         const std::string& bed_file,
                         const alignment::Minimap2Options& options,
                         utils::concurrency::AsyncExecutor&& task_executor)
        : MessageSink(MAX_INPUT_QUEUE_SIZE, 1),
          m_index_for_bam_messages(load_and_get_index(*index_file_access, index_file, options)),
          m_index_file_access(std::move(index_file_access)),
          m_bed_file_access(std::move(bed_file_access)),
          m_task_executor(std::move(task_executor)) {
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
            m_header_sequence_names.emplace_back(entry.sequence_name);
        }
    }
}

AlignerNode::AlignerNode(std::shared_ptr<alignment::IndexFileAccess> index_file_access,
                         std::shared_ptr<alignment::BedFileAccess> bed_file_access,
                         utils::concurrency::AsyncExecutor&& task_executor)
        : MessageSink(MAX_INPUT_QUEUE_SIZE, 1),
          m_index_file_access(std::move(index_file_access)),
          m_bed_file_access(std::move(bed_file_access)),
          m_task_executor(std::move(task_executor)) {}

AlignerNode::~AlignerNode() { stop_input_processing(utils::AsyncQueueTerminateFast::Yes); }

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

utils::HeaderSQRecords AlignerNode::get_sequence_records_for_header() const {
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

template <typename Read>
void AlignerNode::align_read(Read&& read) {
    m_task_executor.send([this, read_ = std::forward<Read>(read)]() mutable {
        thread_local MmTbufPtr tbuf{mm_tbuf_init()};
        align_read_common(read_->read_common, tbuf.get());
        send_message_to_sink(std::move(read_));
    });
}

void AlignerNode::align_bam_message(BamMessage&& bam_message) {
    m_task_executor.send([this, bam_message_ = std::move(bam_message)] {
        thread_local MmTbufPtr tbuf{mm_tbuf_init()};
        auto records = alignment::Minimap2Aligner(m_index_for_bam_messages)
                               .align(bam_message_.data->bam_ptr.get(), tbuf.get());

        bam_message_.data->read_attrs.num_alignments = records.size();
        bam_message_.data->read_attrs.num_secondary_alignments =
                std::count_if(std::begin(records), std::end(records),
                              [](const auto& aln) { return aln->core.flag & BAM_FSECONDARY; });
        bam_message_.data->read_attrs.num_supplementary_alignments =
                std::count_if(std::begin(records), std::end(records),
                              [](const auto& aln) { return aln->core.flag & BAM_FSUPPLEMENTARY; });

        for (auto& record : records) {
            if (m_bedfile_for_bam_messages && !(record->core.flag & BAM_FUNMAP)) {
                auto ref_id = record->core.tid;
                add_bed_hits_to_record(m_header_sequence_names.at(ref_id), record.get());
            }
            auto hts_data = std::make_unique<HtsData>(HtsData{std::move(record),
                                                              bam_message_.data->read_attrs,
                                                              bam_message_.data->barcoding_result});
            send_message_to_sink(BamMessage{std::move(hts_data), bam_message_.client_info});
        }
    });
}

void AlignerNode::input_thread_fn() {
    Message message;
    while (get_input_message(message)) {
        if (message.holds<BamMessage>()) {
            align_bam_message(message.take<BamMessage>());
        } else if (message.holds<SimplexReadPtr>()) {
            align_read(message.take<SimplexReadPtr>());
        } else if (message.holds<DuplexReadPtr>()) {
            align_read(message.take<DuplexReadPtr>());
        } else {
            send_message_to_sink(std::move(message));
        }
    }
}

void AlignerNode::terminate(const TerminateOptions& terminate_options) {
    stop_input_processing(terminate_options.fast);
    m_task_executor.flush();
}

void AlignerNode::restart() {
    start_input_processing([this] { input_thread_fn(); }, "aligner_node");
}

std::string AlignerNode::get_name() const { return "AlignerNode"; }

stats::NamedStats AlignerNode::sample_stats() const {
    stats::NamedStats stats = MessageSink::sample_stats();
    stats.merge(alignment::mm2_limiter_stats());
    stats["queued_tasks"] = double(m_task_executor.tasks_in_flight());
    return stats;
}

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
