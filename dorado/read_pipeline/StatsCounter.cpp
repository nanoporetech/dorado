#include "StatsCounter.h"

#include <spdlog/spdlog.h>

#include <chrono>
#include <sstream>

namespace dorado {

StatsCounterNode::StatsCounterNode(MessageSink& sink, bool duplex)
        : MessageSink(1000),
          m_sink(sink),
          m_num_bases_processed(0),
          m_num_samples_processed(0),
          m_num_reads_processed(0),
          m_initialization_time(std::chrono::system_clock::now()),
          m_duplex(duplex) {
    m_thread = std::make_unique<std::thread>(std::thread(&StatsCounterNode::worker_thread, this));
}

StatsCounterNode::~StatsCounterNode() {
    terminate();
    m_thread->join();
    m_sink.terminate();
}

void StatsCounterNode::worker_thread() {
    Message message;
    while (m_work_queue.try_pop(message)) {
        // If this message isn't a read, we'll get a bad_variant_access exception.
        auto read = std::get<std::shared_ptr<Read>>(message);

        m_num_bases_processed += read->seq.length();
        m_num_samples_processed += read->raw_data.size(0);
        ++m_num_reads_processed;

        m_sink.push_message(read);
    }

    m_sink.terminate();
    m_end_time = std::chrono::system_clock::now();
}

void StatsCounterNode::dump_stats() {
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

}  // namespace dorado
