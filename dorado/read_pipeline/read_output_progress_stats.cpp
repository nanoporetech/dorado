#include "read_output_progress_stats.h"

#include <spdlog/spdlog.h>

#include <cassert>
#include <cmath>
#include <limits>
#include <sstream>

namespace dorado {

namespace {

std::string const PREFIX_PROGRESS_LINE_HDR{"[PROG_STAT_HDR] "};
std::string const PREFIX_PROGRESS_LINE{"[PROG_STAT] "};

struct ReportInfo {
    float time_elapsed;
    float time_remaining;
    std::size_t total_reads_processed;
    std::size_t total_reads_estimate;
    float interval_time_elapsed;
    std::size_t interval_reads_processed;
};

void show_report(const ReportInfo& info) {
    std::ostringstream oss;
    // clang-format off
    oss << PREFIX_PROGRESS_LINE 
        << info.time_elapsed
        << ", " << info.time_remaining
        << ", " << info.total_reads_processed
        << ", " << info.total_reads_estimate
        << ", " << info.interval_time_elapsed
        << ", " << info.interval_reads_processed
        << ", " << 0;
    // clang-format on
    spdlog::info(oss.str());
}

auto get_num_reads_written(const stats::NamedStats& stats) {
    auto size_t_stat = [&stats](const std::string& name) -> std::size_t {
        auto res = stats.find(name);
        if (res != stats.end()) {
            return static_cast<std::size_t>(res->second);
        }
        return 0;
    };

    return size_t_stat("HtsWriter.unique_simplex_reads_written") +
           size_t_stat("BarcodeDemuxerNode.demuxed_reads_written");
}

}  // namespace

ReadOutputProgressStats::ReadOutputProgressStats(std::chrono::seconds interval_duration,
                                                 std::size_t num_input_files,
                                                 StatsCollectionMode stats_collection_mode)
        : m_interval_duration(interval_duration),
          m_num_input_files(num_input_files),
          m_stats_collection_mode(stats_collection_mode),
          m_monitoring_start_time(progress_clock::now()),
          m_interval_start(m_monitoring_start_time) {
    if (is_disabled()) {
        return;
    }
    m_next_report_time = m_monitoring_start_time + m_interval_duration;
    std::ostringstream oss;
    oss << PREFIX_PROGRESS_LINE_HDR << "time elapsed(secs)"
        << ", time remaining (estimate)"
        << ", total reads processed"
        << ", total reads (estimate)"
        << ", interval(secs)"
        << ", interval reads processed"
        << ", interval bases processed";
    spdlog::info(oss.str());
}

void ReadOutputProgressStats::report_final_stats() {
    if (is_disabled()) {
        return;
    }

    report_stats(0, m_last_stats_completed_time);
}

void ReadOutputProgressStats::report_stats(const std::size_t current_reads_written_count,
                                           progress_clock::time_point interval_end) const {
    using namespace std::chrono;
    using Seconds = duration<float>;
    ReportInfo info{};
    info.time_elapsed = duration_cast<Seconds>(interval_end - m_monitoring_start_time).count();
    info.interval_time_elapsed = duration_cast<Seconds>(interval_end - m_interval_start).count();

    info.interval_reads_processed =
            m_interval_previous_stats_total + current_reads_written_count - m_interval_start_count;

    // Total number read from file may be higher than total number written by HtsWriter
    // if input files contained duplicate read_ids. Use the total number written so that
    // the sum of the intervals matches the total number processed.
    info.total_reads_processed = m_previous_stats_total + current_reads_written_count;

    if (is_completed()) {
        info.total_reads_estimate = info.total_reads_processed;
        info.time_remaining = 0.0;

    } else {
        info.total_reads_estimate = get_adjusted_estimated_total_reads(current_reads_written_count);
        if (info.total_reads_processed > 0) {
            auto estimated_total_time =
                    info.time_elapsed *
                    (static_cast<float>(info.total_reads_estimate) / info.total_reads_processed);
            info.time_remaining = estimated_total_time - info.time_elapsed;
        } else {
            info.time_remaining = 0.0;
        }
    }

    show_report(info);
}

void ReadOutputProgressStats::update_stats(const stats::NamedStats& stats) {
    if (is_disabled()) {
        return;
    }
    std::lock_guard lock(m_mutex);

    auto current_time = progress_clock::now();
    if (current_time < m_next_report_time) {
        return;
    }
    auto reads_written = get_num_reads_written(stats);
    report_stats(reads_written, current_time);
    m_interval_start = current_time;
    m_next_report_time += m_interval_duration;
    m_interval_start_count = reads_written;
    m_interval_previous_stats_total = 0;
}

void ReadOutputProgressStats::notify_stats_collector_completed(const stats::NamedStats& stats) {
    if (is_disabled()) {
        return;
    }

    m_last_stats_completed_time = progress_clock::now();
    const auto stats_reads_written = get_num_reads_written(stats);

    std::lock_guard lock(m_mutex);
    // Use m_total_known_readcount instead of calculating from the given stats
    // as m_total_known_readcount will include any duplicate ids missing from the stats.
    m_previous_stats_total += stats_reads_written;
    assert(m_previous_stats_total <= m_total_known_readcount &&
           "Expected that update_reads_per_file_estimate called before "
           "notify_stats_collector_completed");
    // HtsWriter stats don't include any duplicate read_ids, but HtsReader will not filter out duplicate read_ids
    // so account for any discrepancy in this interval
    auto duplicates_read_ids_this_interval =
            static_cast<std::size_t>(m_total_known_readcount - m_previous_stats_total);
    if (duplicates_read_ids_this_interval > 0) {
        m_previous_stats_total += duplicates_read_ids_this_interval;
    }

    m_interval_previous_stats_total +=
            stats_reads_written - m_interval_start_count + duplicates_read_ids_this_interval;
    m_interval_start_count = 0;
}

std::size_t ReadOutputProgressStats::calc_total_reads_single_collector(
        std::size_t current_reads_count) const {
    auto estimated_total = static_cast<std::size_t>(
            std::lrint(m_estimated_num_reads_per_file * m_num_input_files));
    if (current_reads_count <= estimated_total) {
        return estimated_total;
    }

    // Assume we are halfway through
    return current_reads_count * 2;
}

std::size_t ReadOutputProgressStats::calc_total_reads_collector_per_file(
        std::size_t current_reads_count) const {
    if (static_cast<float>(current_reads_count) <= m_estimated_num_reads_per_file) {
        return static_cast<std::size_t>(
                std::lrint(m_estimated_num_reads_per_file * m_num_input_files));
    }

    // Current file exceed reads per file estimate so recalculate assuming we're halfway through the file
    auto assumed_current_file_total_reads = current_reads_count * 2;
    auto total_reads_including_current = m_total_known_readcount + assumed_current_file_total_reads;
    float adjusted_estimated_reads_per_file =
            total_reads_including_current /
            static_cast<float>(m_num_files_where_readcount_known + 1);

    return static_cast<std::size_t>(
            std::lrint(adjusted_estimated_reads_per_file * m_num_input_files));
}

std::size_t ReadOutputProgressStats::get_adjusted_estimated_total_reads(
        std::size_t current_reads_count) const {
    if (m_stats_collection_mode == StatsCollectionMode::single_collector) {
        return calc_total_reads_single_collector(current_reads_count);
    }
    return calc_total_reads_collector_per_file(current_reads_count);
}

void ReadOutputProgressStats::update_reads_per_file_estimate(std::size_t num_reads_in_file) {
    if (is_disabled()) {
        return;
    }

    std::lock_guard lock(m_mutex);
    assert(!is_completed() && "More files updates supplied than input files.");
    ++m_num_files_where_readcount_known;
    m_total_known_readcount += num_reads_in_file;
    m_estimated_num_reads_per_file =
            static_cast<float>(m_total_known_readcount) / m_num_files_where_readcount_known;
}

bool ReadOutputProgressStats::is_completed() const {
    return m_num_files_where_readcount_known == m_num_input_files;
}

bool ReadOutputProgressStats::is_disabled() const {
    return m_interval_duration == std::chrono::seconds{0};
}

}  // namespace dorado
