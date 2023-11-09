#pragma once

#include "utils/stats.h"
#include "utils/tty_utils.h"

#ifdef WIN32
#include <indicators/progress_bar.hpp>
#else
#include <indicators/block_progress_bar.hpp>
#endif
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

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
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(m_end_time -
                                                                              m_initialization_time)
                                .count();
        if (m_num_simplex_reads_written > 0) {
            spdlog::info("> Simplex reads basecalled: {}", m_num_simplex_reads_written);
        }
        if (m_num_simplex_reads_filtered > 0) {
            spdlog::info("> Simplex reads filtered: {}", m_num_simplex_reads_filtered);
        }
        if (m_duplex) {
            spdlog::info("> Duplex reads basecalled: {}", m_num_duplex_reads_written);
            if (m_num_duplex_reads_filtered > 0) {
                spdlog::info("> Duplex reads filtered: {}", m_num_duplex_reads_filtered);
            }
            spdlog::info("> Duplex rate: {}%",
                         ((static_cast<float>(m_num_duplex_bases_processed -
                                              m_num_duplex_bases_filtered) *
                           2) /
                          (m_num_simplex_bases_processed - m_num_simplex_bases_filtered)) *
                                 100);
        }
        if (m_num_bases_processed > 0) {
            std::ostringstream samples_sec;
            if (m_duplex) {
                samples_sec << std::scientific << m_num_bases_processed / (duration / 1000.0);
                spdlog::info("> Basecalled @ Bases/s: {}", samples_sec.str());
            } else {
                samples_sec << std::scientific << m_num_samples_processed / (duration / 1000.0);
                spdlog::info("> Basecalled @ Samples/s: {}", samples_sec.str());
            }
        }

        if (m_num_barcodes_demuxed > 0) {
            std::ostringstream rate_str;
            rate_str << std::scientific << m_num_barcodes_demuxed / (duration / 1000.0);
            spdlog::info("> {} reads demuxed @ classifications/s: {}", m_num_barcodes_demuxed,
                         rate_str.str());
        }
    }

    void update_progress_bar(const stats::NamedStats& stats) {
        // Instead of capturing end time when summarizer is called,
        // which suffers from delays due to sampler and pipeline termination
        // costs, store it whenever stats are updated.
        m_end_time = std::chrono::system_clock::now();

        auto fetch_stat = [&stats](const std::string& name) {
            auto res = stats.find(name);
            if (res != stats.end()) {
                return res->second;
            }
            return 0.;
        };

        m_num_simplex_reads_written = int(fetch_stat("HtsWriter.unique_simplex_reads_written") +
                                          fetch_stat("BarcodeDemuxerNode.demuxed_reads_written"));

        m_num_simplex_reads_filtered = int(fetch_stat("ReadFilterNode.simplex_reads_filtered"));
        m_num_simplex_bases_filtered = int(fetch_stat("ReadFilterNode.simplex_bases_filtered"));
        m_num_simplex_bases_processed = int64_t(fetch_stat("BasecallerNode.bases_processed"));
        m_num_bases_processed = m_num_simplex_bases_processed;
        m_num_samples_processed = int64_t(fetch_stat("BasecallerNode.samples_processed"));
        if (m_duplex) {
            m_num_duplex_bases_processed =
                    int64_t(fetch_stat("StereoBasecallerNode.bases_processed"));
            m_num_bases_processed += m_num_duplex_bases_processed;
            m_num_samples_processed +=
                    int64_t(fetch_stat("StereoBasecallerNode.samples_processed"));
        }
        m_num_duplex_reads_written = int(fetch_stat("HtsWriter.duplex_reads_written"));
        m_num_duplex_reads_filtered = int(fetch_stat("ReadFilterNode.duplex_reads_filtered"));
        m_num_duplex_bases_filtered = int(fetch_stat("ReadFilterNode.duplex_bases_filtered"));

        // Barcode demuxing stats.
        m_num_barcodes_demuxed = int(fetch_stat("BarcodeClassifierNode.num_barcodes_demuxed"));

        // don't output progress bar if stderr is not a tty
        if (!utils::is_fd_tty(stderr)) {
            return;
        }

        if (m_num_reads_expected != 0) {
            // TODO: Add the ceiling because in duplex, reads written can exceed reads expected
            // because of the read splitting. That needs to be handled properly.
            float progress =
                    std::min(100.f, 100.f *
                                            static_cast<float>(m_num_simplex_reads_written +
                                                               m_num_simplex_reads_filtered) /
                                            m_num_reads_expected);
            if (progress > 0 && progress > m_last_progress_written) {
                m_progress_bar.set_progress(size_t(progress));
#ifndef WIN32
                std::cerr << "\033[K";
#endif  // WIN32
                m_last_progress_written = progress;
                std::cerr << "\r";
            }
        } else {
            std::cerr << "\r> Output records written: " << m_num_simplex_reads_written;
            std::cerr << "\r";
        }
    }

private:
    int64_t m_num_bases_processed{0};
    int64_t m_num_samples_processed{0};
    int64_t m_num_simplex_bases_processed{0};
    int64_t m_num_duplex_bases_processed{0};
    int m_num_simplex_reads_written{0};
    int m_num_simplex_reads_filtered{0};
    int m_num_simplex_bases_filtered{0};
    int m_num_duplex_reads_written{0};
    int m_num_duplex_reads_filtered{0};
    int m_num_duplex_bases_filtered{0};
    int m_num_barcodes_demuxed{0};

    int m_num_reads_expected;

    std::chrono::time_point<std::chrono::system_clock> m_initialization_time;
    std::chrono::time_point<std::chrono::system_clock> m_end_time;

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

    float m_last_progress_written = -1.f;
};

}  // namespace dorado
