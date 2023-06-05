#include "StatsCounter.h"

#include <spdlog/spdlog.h>

#include <chrono>
#include <sstream>
#include <thread>

namespace dorado {

StatsCounter::StatsCounter(int total_reads, bool duplex)
        : m_num_bases_processed(0),
          m_num_samples_processed(0),
          m_num_reads_processed(0),
          m_num_reads_filtered(0),
          m_num_reads_expected(total_reads),
          m_initialization_time(std::chrono::system_clock::now()),
          m_duplex(duplex) {
    m_thread = std::make_unique<std::thread>(std::thread(&StatsCounter::worker_thread, this));
}

void StatsCounter::add_basecalled_read(std::shared_ptr<Read> read) {
    m_num_bases_processed += read->seq.length();
    m_num_samples_processed += read->raw_data.size(0);
    ++m_num_reads_processed;
}

void StatsCounter::add_written_read_id(const std::string& read_id) {
    const std::lock_guard<std::mutex> lock(m_reads_mutex);
    m_processed_read_ids.emplace(std::move(read_id));
}

void StatsCounter::add_filtered_read_id(const std::string& read_id) {
    const std::lock_guard<std::mutex> lock(m_reads_mutex);
    m_processed_read_ids.emplace(std::move(read_id));
}

void StatsCounter::worker_thread() {
    size_t write_count = 0;
    float m_last_progress_written = -1.f;

    bool bar_initialized = false;
    while (!m_terminate.load()) {
        if (m_num_reads_expected != 0 && !bar_initialized) {
            m_progress_bar.set_progress(0.f);
            bar_initialized = true;
        }

        {
            const std::lock_guard<std::mutex> lock(m_reads_mutex);
            write_count = m_processed_read_ids.size();
        }

        if (m_num_reads_expected != 0) {
            float progress = 100.f * static_cast<float>(write_count) / m_num_reads_expected;
            if (progress > m_last_progress_written) {
                m_progress_bar.set_progress(progress);
#ifndef WIN32
                std::cerr << "\033[K";
#endif  // WIN32
                m_last_progress_written = progress;
                std::cerr << "\r";
            }
        } else {
            std::cerr << "\r> Output records written: " << write_count;
            std::cerr << "\r";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void StatsCounter::dump_stats() {
    m_terminate = true;
    m_thread->join();

    m_end_time = std::chrono::system_clock::now();
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
