#include "ReadToBamTypeNode.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>

namespace dorado {

void ReadToBamType::worker_thread() {
    torch::InferenceMode inference_mode_guard;

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
                             size_t max_reads)
        : MessageSink(max_reads),
          m_num_worker_threads(num_worker_threads),
          m_emit_moves(emit_moves),
          m_modbase_threshold(
                  static_cast<uint8_t>(std::min(modbase_threshold_frac * 256.0f, 255.0f))) {
    start_threads();
}

void ReadToBamType::start_threads() {
    m_num_worker_threads = 1;
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
