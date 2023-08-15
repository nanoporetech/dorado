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
        if (!std::holds_alternative<std::shared_ptr<Read>>(message)) {
            send_message_to_sink(std::move(message));
            continue;
        }

        // If this message isn't a read, we'll get a bad_variant_access exception.
        auto read = std::get<std::shared_ptr<Read>>(message);

        if (m_rna) {
            std::reverse(read->seq.begin(), read->seq.end());
            std::reverse(read->qstring.begin(), read->qstring.end());
        }

        auto alns = read->extract_sam_lines(m_emit_moves, m_modbase_threshold);
        for (auto& aln : alns) {
            send_message_to_sink(std::move(aln));
        }
    }
}

ReadToBamType::ReadToBamType(bool emit_moves,
                             bool rna,
                             size_t num_worker_threads,
                             float modbase_threshold_frac,
                             size_t max_reads)
        : MessageSink(max_reads),
          m_num_worker_threads(num_worker_threads),
          m_emit_moves(emit_moves),
          m_rna(rna),
          m_modbase_threshold(
                  static_cast<uint8_t>(std::min(modbase_threshold_frac * 256.0f, 255.0f))) {
    start_threads();
}

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
