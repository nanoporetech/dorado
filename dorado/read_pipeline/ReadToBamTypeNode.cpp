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
            send_message_to_sink(BamMessage{std::move(aln), read_common_data.client_info});
        }
    }
}

ReadToBamTypeNode::ReadToBamTypeNode(bool emit_moves,
                                     size_t num_worker_threads,
                                     float modbase_threshold_frac,
                                     std::unique_ptr<const utils::SampleSheet> sample_sheet,
                                     size_t max_reads)
        : MessageSink(max_reads, static_cast<int>(num_worker_threads)),
          m_emit_moves(emit_moves),
          m_modbase_threshold(
                  static_cast<uint8_t>(std::min(modbase_threshold_frac * 256.0f, 255.0f))),
          m_sample_sheet(std::move(sample_sheet)) {
    start_input_processing(&ReadToBamTypeNode::input_thread_fn, this);
}

stats::NamedStats ReadToBamTypeNode::sample_stats() const { return stats::from_obj(m_work_queue); }

}  // namespace dorado
