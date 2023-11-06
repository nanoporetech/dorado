#include "AlignerNode.h"

#include "alignment/Minimap2Aligner.h"
#include "alignment/Minimap2Index.h"

#include <minimap.h>
#include <spdlog/spdlog.h>

#include <cassert>
#include <filesystem>
#include <string>
#include <vector>

namespace dorado {

AlignerNode::AlignerNode(std::shared_ptr<alignment::IndexFileAccess> index_file_access,
                         const std::string& filename,
                         const alignment::Minimap2Options& options,
                         int threads)
        : MessageSink(10000),
          m_threads(threads),
          m_index_file_access(std::move(index_file_access)) {
    set_bam_index(filename, options, threads);
    start_threads();
}

AlignerNode::AlignerNode(std::shared_ptr<alignment::IndexFileAccess> index_file_access, int threads)
        : MessageSink(10000),
          m_threads(threads),
          m_index_file_access(std::move(index_file_access)) {
    start_threads();
}

void AlignerNode::set_bam_index(const std::string& filename,
                                const alignment::Minimap2Options& options,
                                int threads) {
    int num_index_construction_threads{options.print_aln_seq ? 1 : static_cast<int>(m_threads)};
    switch (m_index_file_access->load_index(filename, options, num_index_construction_threads)) {
    case alignment::IndexLoadResult::reference_file_not_found:
        throw std::runtime_error("AlignerNode reference path does not exist: " + filename);
    case alignment::IndexLoadResult::validation_error:
        throw std::runtime_error("AlignerNode validation error checking minimap options");
    case alignment::IndexLoadResult::split_index_not_supported:
        throw std::runtime_error(
                "Dorado doesn't support split index for alignment. Please re-run with larger index "
                "size.");
    case alignment::IndexLoadResult::success:
        break;
    }
    m_bam_index = m_index_file_access->get_index(filename, options);
}

std::shared_ptr<alignment::Minimap2Index> AlignerNode::get_index(const ReadCommon& read_common) {
    auto& align_info = read_common.client_access->alignment_info();
    if (align_info.reference_file.empty()) {
        return {};
    }
    auto index =
            m_index_file_access->get_index(align_info.reference_file, align_info.minimap_options);
    assert(index != nullptr, "Expect an index file to be loaded for a specifieic reference");
    return index;
}

void AlignerNode::start_threads() {
    for (size_t i = 0; i < m_threads; i++) {
        m_workers.push_back(std::thread(&AlignerNode::worker_thread, this));
    }
}

void AlignerNode::terminate_impl() {
    terminate_input_queue();
    for (auto& m : m_workers) {
        if (m.joinable()) {
            m.join();
        }
    }
    m_workers.clear();
}

void AlignerNode::restart() {
    restart_input_queue();
    start_threads();
}

AlignerNode::~AlignerNode() { terminate_impl(); }

alignment::HeaderSquenceRecords AlignerNode::get_sequence_records_for_header() const {
    assert(m_bam_index != nullptr &&
           "get_sequence_records_for_header only valid if AlignerNode constructed with index file");
    return alignment::Minimap2Aligner(m_bam_index).get_sequence_records_for_header();
}

void AlignerNode::worker_thread() {
    Message message;
    mm_tbuf_t* tbuf = mm_tbuf_init();
    while (get_input_message(message)) {
        if (std::holds_alternative<BamPtr>(message)) {
            auto read = std::get<BamPtr>(std::move(message));
            auto records = alignment::Minimap2Aligner(m_bam_index).align(read.get(), tbuf);
            for (auto& record : records) {
                send_message_to_sink(std::move(record));
            }
        } else if (std::holds_alternative<SimplexReadPtr>(message)) {
            auto read = std::get<SimplexReadPtr>(std::move(message));
            auto index = get_index(read->read_common);
            if (index) {
                alignment::Minimap2Aligner(index).align(*read, tbuf);
            }
            send_message_to_sink(std::move(read));
        } else {
            send_message_to_sink(std::move(message));
            continue;
        }
    }
    mm_tbuf_destroy(tbuf);
}

stats::NamedStats AlignerNode::sample_stats() const { return stats::from_obj(m_work_queue); }

}  // namespace dorado
