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
    long double time_elapsed;
    long double time_remaining;
    std::size_t total_reads_processed;
    std::size_t total_reads_estimate;
    long double interval_time_elapsed;
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
    auto size_t_stat = [&stats](const std::string& name) {
        auto res = stats.find(name);
        if (res != stats.end()) {
            return static_cast<std::size_t>(res->second);
        }
        return std::size_t{};
    };

    return size_t_stat("HtsWriter.unique_simplex_reads_written") +
           size_t_stat("BarcodeDemuxerNode.demuxed_reads_written");
}

}  // namespace

ReadOutputProgressStats::ReadOutputProgressStats(std::chrono::seconds interval_duration,
                                                 std::size_t num_input_files)
        : m_interval_duration(interval_duration),
          m_num_input_files(num_input_files),
          m_monitoring_start_time(progress_clock::now()),
          m_interval_start(m_monitoring_start_time),
          m_next_report_time(m_monitoring_start_time + m_interval_duration) {
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

ReadOutputProgressStats::~ReadOutputProgressStats() {
    auto current_time = progress_clock::now();
    m_interval_end = m_last_stats_completed_time;
    report_stats(0);
}

void ReadOutputProgressStats::report_stats(const std::size_t current_reads_written_count) {
    using namespace std::chrono;
    ReportInfo info{};
    info.time_elapsed =
            duration_cast<duration<long double>>(m_interval_end - m_monitoring_start_time).count();
    info.interval_time_elapsed =
            duration_cast<duration<long double>>(m_interval_end - m_interval_start).count();

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
    std::lock_guard lock(m_mutex);

    auto current_time = progress_clock::now();
    if (current_time < m_next_report_time) {
        return;
    }
    auto reads_written = get_num_reads_written(stats);
    m_interval_end = current_time;
    report_stats(reads_written);
    m_interval_start = current_time;
    m_next_report_time += +m_interval_duration;
    m_interval_start_count = reads_written;
    m_interval_previous_stats_total = 0;
}

void ReadOutputProgressStats::notify_stats_completed(const stats::NamedStats& stats) {
    m_last_stats_completed_time = progress_clock::now();
    const auto stats_reads_written = get_num_reads_written(stats);

    std::lock_guard lock(m_mutex);
    m_previous_stats_total += stats_reads_written;
    m_interval_previous_stats_total += stats_reads_written - m_interval_start_count;
    m_interval_start_count = 0;
}

std::size_t ReadOutputProgressStats::get_adjusted_estimated_total_reads(
        std::size_t current_reads_count) {
    if (static_cast<float>(current_reads_count) <= m_estimated_num_reads_per_file) {
        return static_cast<std::size_t>(lround(m_estimated_num_reads_per_file * m_num_input_files));
    }

    // Current file exceed reads per file estimate so recalculate assuming we're halfway through the file
    auto assumed_current_file_total_reads = current_reads_count * 2;
    auto total_reads_including_current = m_total_known_readcount + assumed_current_file_total_reads;
    float adjusted_estimated_reads_per_file =
            total_reads_including_current /
            static_cast<float>(m_num_files_where_readcount_known + 1);

    return static_cast<std::size_t>(lround(adjusted_estimated_reads_per_file * m_num_input_files));
}

void ReadOutputProgressStats::update_reads_per_file_estimate(std::size_t num_reads_in_file) {
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

}  // namespace dorado
