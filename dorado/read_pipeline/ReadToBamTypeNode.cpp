#include "ReadToBamTypeNode.h"

#include "utils/SampleSheet.h"

#include <spdlog/spdlog.h>

#include <algorithm>

namespace dorado {

void ReadToBamTypeNode::input_thread_fn() {
    at::InferenceMode inference_mode_guard;

    Message message;
    while (get_input_message(message)) {
        // If this message isn't a read, just forward it to the sink.
        if (!is_read_message(message)) {
            send_message_to_sink(std::move(message));
            continue;
        }

        auto& read_common_data = get_read_common_data(message);

        bool is_duplex_parent = false;
        if (!read_common_data.is_duplex) {
            is_duplex_parent = std::get<SimplexReadPtr>(message)->is_duplex_parent;
        }

        // alias barcode if present
        if (m_sample_sheet && !read_common_data.barcode.empty()) {
            auto alias = m_sample_sheet->get_alias(
                    read_common_data.flowcell_id, read_common_data.position_id,
                    read_common_data.experiment_id, read_common_data.barcode);
            if (!alias.empty()) {
                read_common_data.barcode = alias;
            }
        }

        auto alns = read_common_data.extract_sam_lines(m_emit_moves, m_modbase_threshold,
                                                       is_duplex_parent);
        for (auto& aln : alns) {
            BamMessage bam_msg{std::move(aln), read_common_data.client_info};
            bam_msg.sequencing_kit = read_common_data.sequencing_kit;
            send_message_to_sink(std::move(bam_msg));
        }
    }
}

ReadToBamTypeNode::ReadToBamTypeNode(bool emit_moves,
                                     size_t num_worker_threads,
                                     std::optional<float> modbase_threshold_frac,
                                     std::unique_ptr<const utils::SampleSheet> sample_sheet,
                                     size_t max_reads)
        : MessageSink(max_reads, static_cast<int>(num_worker_threads)),
          m_emit_moves(emit_moves),
          m_sample_sheet(std::move(sample_sheet)) {
    if (modbase_threshold_frac) {
        set_modbase_threshold(*modbase_threshold_frac);
    }
}

ReadToBamTypeNode::~ReadToBamTypeNode() { stop_input_processing(); }

stats::NamedStats ReadToBamTypeNode::sample_stats() const { return stats::from_obj(m_work_queue); }

void ReadToBamTypeNode::set_modbase_threshold(float threshold) {
    if (threshold < 0.f || threshold > 1.f) {
        throw std::runtime_error("modbase threshold must be between 0 and 1.");
    }
    m_modbase_threshold = static_cast<uint8_t>(std::min(threshold * 256.0f, 255.0f));
}

}  // namespace dorado
