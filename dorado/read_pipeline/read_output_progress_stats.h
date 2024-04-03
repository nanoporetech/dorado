#pragma once

#include "utils/stats.h"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <thread>

namespace dorado {

class ReadOutputProgressStats {
public:
    enum class StatsCollectionMode {
        single_collector,  // demux has a single pipeline into which all input files are passed
        collector_per_input_file,  // aligner creates a new pipeline for each input file
    };

private:
    using progress_clock = std::chrono::steady_clock;
    const std::chrono::seconds m_interval_duration;
    const std::size_t m_num_input_files;
    const StatsCollectionMode m_stats_collection_mode;

    std::mutex m_mutex;
    progress_clock::time_point m_monitoring_start_time;
    progress_clock::time_point m_interval_start;
    progress_clock::time_point m_next_report_time;

    std::size_t m_previous_stat_collectors_total{};
    std::size_t m_interval_previous_stat_collectors_total{};
    std::size_t m_interval_start_count{};
    std::size_t m_current_reads_written_count{};

    struct StatsForPostProcessing {
        std::size_t interval_reads_processed{};
        std::size_t total_reads_processed{};
        std::size_t total_reads_estimate{};
    };
    std::optional<StatsForPostProcessing> m_post_processing_stats{};

    std::size_t m_num_files_where_readcount_known{};

    // incremented in after post processing is completed only relevant if
    // StatsCollectionMode is collector_per_input_file.
    // Used to identify the interval range for percentage complete of post processing
    std::size_t m_num_completed_files{};

    std::size_t m_total_known_readcount{};
    float m_estimated_num_reads_per_file{};
    float m_post_processing_percentage{};
    float m_post_processing_progress{};

    bool m_is_finished{false};
    bool m_report_final_stats{false};
    std::condition_variable m_stop{};
    std::thread m_reporting_thread{};

    void report_stats(progress_clock::time_point interval_end) const;

    std::size_t calc_total_reads_single_collector(std::size_t current_reads_count) const;
    std::size_t calc_total_reads_collector_per_file(std::size_t current_reads_count) const;
    std::size_t get_adjusted_estimated_total_reads(std::size_t current_reads_count) const;
    std::pair<float, float> get_current_range() const;

    bool is_known_total_number_input_reads() const;
    bool is_disabled() const;
    void join_report_thread();

public:
    ReadOutputProgressStats(std::chrono::seconds interval_duration,
                            std::size_t num_input_files,
                            StatsCollectionMode stats_collection_mode);

    ~ReadOutputProgressStats();

    void start();

    void update_stats(const stats::NamedStats& stats);

    // Called to indicate the current stats collection has completed.
    // There may be new stats but their counters will be reset to zero.
    // E.g. happens in `dorado aligner` when reading from multiple input files
    // a new pipeline and HtsWriter is created for each input file
    void notify_stats_collector_completed(const stats::NamedStats& stats);

    // Useful for collector_per_input_file mode (aligner), so that further
    // report outputs will be based on read stats instead of post processing
    void notify_post_processing_completed();

    void update_reads_per_file_estimate(std::size_t num_reads_in_file);

    void set_post_processing_percentage(float post_processing_percentage);

    void update_post_processing_progress(float progress);

    void report_final_stats();
};

}  // namespace dorado
