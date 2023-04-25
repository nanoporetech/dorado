#include "ReadToBamTypeNode.h"

#include <spdlog/spdlog.h>

#include <chrono>

namespace dorado {

void ReadToBamType::worker_thread() {
    m_active_threads++;

    Message message;
    while (m_work_queue.try_pop(message)) {
        // If this message isn't a read, we'll get a bad_variant_access exception.
        auto read = std::get<std::shared_ptr<Read>>(message);

        m_num_bases_processed += read->seq.length();
        m_num_samples_processed += read->raw_data.size(0);
        ++m_num_reads_processed;

        if (m_rna) {
            std::reverse(read->seq.begin(), read->seq.end());
            std::reverse(read->qstring.begin(), read->qstring.end());
        }

        auto alns = read->extract_sam_lines(m_emit_moves, m_duplex, m_modbase_threshold);
        for (auto aln : alns) {
            m_sink.push_message(aln);
        }
    }

    auto num_active_threads = --m_active_threads;
    if (num_active_threads == 0) {
        m_sink.terminate();

        m_end_time = std::chrono::system_clock::now();
    }
}

void ReadToBamType::dump_stats() {
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(m_end_time -
                                                                          m_initialization_time)
                            .count();
    std::ostringstream samples_sec;
    spdlog::info("> Reads basecalled: {}", m_num_reads_processed);
    if (m_duplex) {
        samples_sec << std::scientific << m_num_bases_processed / (duration / 1000.0);
        spdlog::info("> Basecalled @ Bases/s: {}", samples_sec.str());
    } else {
        samples_sec << std::scientific << m_num_samples_processed / (duration / 1000.0);
        spdlog::info("> Basecalled @ Samples/s: {}", samples_sec.str());
    }
}

ReadToBamType::ReadToBamType(MessageSink& sink,
                             bool emit_moves,
                             bool rna,
                             bool duplex,
                             size_t num_worker_threads,
                             uint8_t modbase_threshold,
                             size_t max_reads)
        : MessageSink(max_reads),
          m_sink(sink),
          m_emit_moves(emit_moves),
          m_rna(rna),
          m_duplex(duplex),
          m_modbase_threshold(modbase_threshold),
          m_active_threads(0),
          m_num_bases_processed(0),
          m_num_samples_processed(0),
          m_initialization_time(std::chrono::system_clock::now()) {
    for (size_t i = 0; i < num_worker_threads; i++) {
        m_workers.push_back(
                std::make_unique<std::thread>(std::thread(&ReadToBamType::worker_thread, this)));
    }
}

ReadToBamType::~ReadToBamType() {
    terminate();
    for (auto& m : m_workers) {
        m->join();
    }
    m_sink.terminate();
}

}  // namespace dorado
