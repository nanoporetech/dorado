#pragma once

#include "SimpleProgressBar.h"
#include "utils/stats.h"

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
    enum Mode : uint8_t { SIMPLEX, DUPLEX, DEMUX, TRIM, ALIGN };

    ProgressTracker(Mode mode, int total_reads);
    ProgressTracker(Mode mode, int total_reads, float post_processing_percentage);
    ~ProgressTracker();

    void set_description(const std::string& desc);
    void set_total_reads(int num_reads) { m_num_reads_expected = num_reads; }
    void set_post_processing_percentage(float pct) { m_post_processing_percentage = pct; }

    void mark_as_completed();
    void summarize() const;
    void update_progress_bar(const stats::NamedStats& stats);
    void update_post_processing_progress(float progress);

    void reset_initialization_time();

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

    int64_t m_total_records_written{0};
    int64_t m_primary_records_written{0};
    int64_t m_unmapped_records_written{0};
    int64_t m_secondary_records_written{0};
    int64_t m_supplementary_records_written{0};

    int m_num_barcodes_demuxed{0};
    int m_num_midstrand_barcodes{0};
    int m_num_poly_a_called{0};
    int m_num_poly_a_not_called{0};
    int m_avg_poly_a_tail_lengths{0};
    int m_num_untrimmed_short_reads{0};

    int64_t m_num_mods_samples_processed{0};
    int64_t m_num_mods_samples_incl_padding{0};

    int m_num_reads_expected;

    std::map<std::string, size_t> m_barcode_count;
    std::map<int, int> m_poly_a_tail_length_count;

    std::chrono::time_point<std::chrono::system_clock> m_initialization_time;
    std::chrono::time_point<std::chrono::system_clock> m_end_time;

    const Mode m_mode;

    SimpleProgressBar m_progress_bar;

    float m_last_progress_written = -1.f;

    // What % of time is going to be spent in post-processing.
    float m_post_processing_percentage;
    float m_last_post_processing_progress = -1.f;

    bool m_is_progress_reporting_disabled{false};
};

}  // namespace dorado
