#pragma once

#include "utils/stats.h"

#ifdef _WIN32
#include <indicators/progress_bar.hpp>
#else
#include <indicators/block_progress_bar.hpp>
#endif

#include <chrono>
#include <cstdint>
#include <map>
#include <string>

namespace dorado {

// Collect and calculate throughput related
// statistics for the pipeline to track dorado
// overall performance.
class ProgressTracker {
public:
    enum Mode : uint8_t { SIMPLEX, DUPLEX, TRIM, ALIGN };

    ProgressTracker(Mode mode, int total_reads, float post_processing_percentage);
    ~ProgressTracker();

    void set_description(const std::string& desc);

    void summarize() const;
    void update_progress_bar(const stats::NamedStats& stats);
    void update_post_processing_progress(float progress);

    // Disable the update of progress information during processing
    // This is useful since progress is not reported using spdlog
    // so it may interleave.
    // Note, summarize will not be disabled as it uses spdlog to report.
    void disable_progress_reporting();

private:
    void internal_set_progress(float progress, bool post_processing);

private:
    int64_t m_num_bases_processed{0};
    int64_t m_num_samples_processed{0};
    int64_t m_num_samples_incl_padding{0};
    int64_t m_num_simplex_bases_processed{0};
    int64_t m_num_duplex_bases_processed{0};
    int m_num_simplex_reads_written{0};
    int m_num_simplex_reads_filtered{0};
    int m_num_simplex_bases_filtered{0};
    int m_num_duplex_reads_written{0};
    int m_num_duplex_reads_filtered{0};
    int m_num_duplex_bases_filtered{0};
    int m_num_barcodes_demuxed{0};
    int m_num_poly_a_called{0};
    int m_num_poly_a_not_called{0};
    int m_avg_poly_a_tail_lengths{0};
    int m_num_untrimmed_short_reads{0};

    int64_t m_num_mods_samples_processed{0};
    int64_t m_num_mods_samples_incl_padding{0};

    const int m_num_reads_expected;

    std::map<std::string, size_t> m_barcode_count;
    std::map<int, int> m_poly_a_tail_length_count;

    std::chrono::time_point<std::chrono::system_clock> m_initialization_time;
    std::chrono::time_point<std::chrono::system_clock> m_end_time;

    const Mode m_mode;

#ifdef _WIN32
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

    // What % of time is going to be spent in post-processing.
    const float m_post_processing_percentage;
    float m_last_post_processing_progress = -1.f;

    bool m_is_progress_reporting_disabled{false};
};

}  // namespace dorado
