#include "ReadToBamTypeNode.h"

#include "utils/SampleSheet.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>

namespace dorado {

void ReadToBamType::worker_thread() {
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
            send_message_to_sink(std::move(aln));
        }
    }
}

ReadToBamType::ReadToBamType(bool emit_moves,
                             size_t num_worker_threads,
                             float modbase_threshold_frac,
                             std::unique_ptr<const utils::SampleSheet> sample_sheet,
                             size_t max_reads)
        : MessageSink(max_reads),
          m_num_worker_threads(num_worker_threads),
          m_emit_moves(emit_moves),
          m_modbase_threshold(
                  static_cast<uint8_t>(std::min(modbase_threshold_frac * 256.0f, 255.0f))),
          m_sample_sheet(std::move(sample_sheet)) {
    start_threads();
}

ReadToBamType::~ReadToBamType() { terminate_impl(); }

void ReadToBamType::start_threads() {
    for (size_t i = 0; i < m_num_worker_threads; i++) {
        m_workers.push_back(
                std::make_unique<std::thread>(std::thread(&ReadToBamType::worker_thread, this)));
    }
}

void ReadToBamType::terminate_impl() {
    terminate_input_queue();
    for (auto& m : m_workers) {
        if (m->joinable()) {
            m->join();
        }
    }
    m_workers.clear();
}

void ReadToBamType::restart() {
    restart_input_queue();
    start_threads();
}

}  // namespace dorado
