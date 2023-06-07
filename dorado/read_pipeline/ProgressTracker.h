#pragma once

#include "ReadPipeline.h"
#include "utils/stats.h"

#ifdef WIN32
#include <indicators/progress_bar.hpp>
#else
#include <indicators/block_progress_bar.hpp>
#endif
#include <spdlog/spdlog.h>

#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>

namespace dorado {

// Collect and calculate throughput related
// statistics for the pipeline to track dorado
// overall performance.
class ProgressTracker {
public:
    ProgressTracker(int total_reads, bool duplex)
            : m_num_reads_expected(total_reads), m_duplex(duplex) {
        m_initialization_time = std::chrono::system_clock::now();
    }

    ~ProgressTracker() = default;

    void summarize() const {
        auto m_end_time = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(m_end_time -
                                                                              m_initialization_time)
                                .count();
        if (m_num_reads_processed > 0) {
            std::ostringstream samples_sec;
            spdlog::info("> Reads basecalled: {}", m_num_reads_written);
            spdlog::info("> Reads filtered: {}", m_num_reads_filtered);
            if (m_duplex) {
                samples_sec << std::scientific << m_num_bases_processed / (duration / 1000.0);
                spdlog::info("> Basecalled @ Bases/s: {}", samples_sec.str());
            } else {
                samples_sec << std::scientific << m_num_samples_processed / (duration / 1000.0);
                spdlog::info("> Basecalled @ Samples/s: {}", samples_sec.str());
            }
        }
    }

    void update_progress_bar(const stats::NamedStats& stats) {
        //for (const auto& [name, value] : stats)
        //{
        //    std::cerr << name << std::endl;
        //}
        if (m_num_reads_expected != 0 && !m_bar_initialized) {
            m_progress_bar.set_progress(0.f);
            m_bar_initialized = true;
        }

        m_num_reads_written = stats.at("HtsWriter.unique_simplex_reads_written");
        // TODO: Clean up how stats are captured from the stats object. Not all
        // stats are available in all runs, so more error handling is needed.
        // The current conditions take care of basecaling vs alignment runs,
        // but this may not be sufficient in the future.
        if (m_num_reads_expected != 0) {
            m_num_reads_filtered = stats.at("ReadFilterNode.reads_filtered");
            m_num_bases_processed = stats.at("BasecallerNode.bases_processed");
            m_num_samples_processed = stats.at("BasecallerNode.samples_processed");
            if (m_duplex) {
                auto res = stats.find("StereoBasecallerNode.bases_processed");
                if (res != stats.end()) {
                    m_num_bases_processed += res->second;
                    m_num_samples_processed = stats.at("StereoBasecallerNode.samples_processed");
                }
            }

            float progress = 100.f *
                             static_cast<float>(m_num_reads_written + m_num_reads_filtered) /
                             m_num_reads_expected;
            if (progress > m_last_progress_written) {
                m_progress_bar.set_progress(progress);
#ifndef WIN32
                std::cerr << "\033[K";
#endif  // WIN32
                m_last_progress_written = progress;
            }
        } else {
            std::cerr << "\r> Output records written: " << m_num_reads_written;
        }
        std::cerr << "\r";
    }

private:
    void worker_thread();

    // Async worker for writing.
    std::unique_ptr<std::thread> m_thread;

    std::atomic<int64_t> m_num_bases_processed;
    std::atomic<int64_t> m_num_samples_processed;
    std::atomic<int> m_num_reads_processed;
    std::atomic<int> m_num_reads_written;
    std::atomic<int> m_num_reads_filtered;

    int m_num_reads_expected;
    bool m_bar_initialized = false;

    std::chrono::time_point<std::chrono::system_clock> m_initialization_time;

    bool m_duplex;

#ifdef WIN32
    indicators::ProgressBar m_progress_bar {
#else
    indicators::BlockProgressBar m_progress_bar{
#endif
        indicators::option::Stream{std::cerr}, indicators::option::BarWidth{30},
                indicators::option::ShowElapsedTime{true},
                indicators::option::ShowRemainingTime{true},
                indicators::option::ShowPercentage{true},
    };

    std::atomic<bool> m_terminate{false};

    std::unordered_set<std::string> m_processed_read_ids;
    float m_last_progress_written = -1.f;

    std::mutex m_reads_mutex;
};

}  // namespace dorado
