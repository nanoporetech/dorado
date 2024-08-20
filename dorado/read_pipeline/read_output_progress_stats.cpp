#include "read_output_progress_stats.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <sstream>

namespace dorado {

namespace {

const std::string PREFIX_PROGRESS_LINE_HDR{"[PROG_STAT_HDR] "};
const std::string PREFIX_PROGRESS_LINE{"[PROG_STAT] "};

// Assuming we are 3/4 of the way through the current give a smoother change
// to estimated total number of reads than assuming half.
constexpr float ASSUMED_PERCENTAGE_THROUGH_INPUT_FILE{0.75};

struct ReportInfo {
    float time_elapsed;
    float time_remaining;
    std::size_t total_reads_processed;
    std::size_t total_reads_estimate;
    float interval_time_elapsed;
    std::size_t interval_reads_processed;
    std::size_t estimated_percentage;
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
        << ", " << 0
        << ", " << info.estimated_percentage;
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

std::size_t to_size_t(double value) { return static_cast<std::size_t>(std::llround(value)); }

}  // namespace

ReadOutputProgressStats::ReadOutputProgressStats(std::chrono::seconds interval_duration,
                                                 std::size_t num_input_files,
                                                 StatsCollectionMode stats_collection_mode)
        : m_interval_duration(interval_duration),
          m_num_input_files(num_input_files),
          m_stats_collection_mode(stats_collection_mode) {
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
        << ", interval bases processed"
        << ", estimated %";
    spdlog::info(oss.str());
}

ReadOutputProgressStats::~ReadOutputProgressStats() {
    {
        std::lock_guard lock(m_mutex);
        m_is_finished = true;
    }
    join_report_thread();
}

void ReadOutputProgressStats::join_report_thread() {
    m_stop.notify_all();

    if (m_reporting_thread.joinable()) {
        m_reporting_thread.join();
    }
}

void ReadOutputProgressStats::start() {
    if (is_disabled()) {
        return;
    }

    m_reporting_thread = std::thread([this] {
        std::unique_lock lock(m_mutex);
        m_monitoring_start_time = progress_clock::now();
        m_interval_start = m_monitoring_start_time;
        auto next_report_time = m_monitoring_start_time + m_interval_duration;
        while (!m_stop.wait_until(lock, next_report_time,
                                  [this] { return m_is_finished || m_report_final_stats; })) {
            auto current_time = progress_clock::now();
            report_stats(current_time);
            if (m_post_processing_stats) {
                m_post_processing_stats->interval_reads_processed =
                        0;  // reset so that not included in next interval
            }

            m_interval_start = current_time;
            m_interval_start_count = m_current_reads_written_count;
            m_interval_previous_stat_collectors_total = 0;

            next_report_time += m_interval_duration;
        }
        if (m_report_final_stats) {
            report_stats(progress_clock::now());
        }
    });
}

void ReadOutputProgressStats::report_final_stats() {
    if (is_disabled()) {
        return;
    }
    {
        std::unique_lock lock(m_mutex);
        // We have been informed that these are final stats so it is safe
        // to set m_current_reads_written_count to zero here without worrying
        // whether another thread will update it before the final stats are
        // actually written.
        m_current_reads_written_count = 0;
        m_report_final_stats = true;
    }
    join_report_thread();
}

std::pair<float, float> ReadOutputProgressStats::get_current_range() const {
    if (m_stats_collection_mode == StatsCollectionMode::single_collector) {
        return {0.0f, 1.0f};
    }
    if (m_num_completed_files == m_num_input_files) {
        return {1.0f, 1.0f};
    }
    float interval_per_file = 1.0f / m_num_input_files;
    float from = m_num_completed_files * interval_per_file;
    float to = std::min(from + interval_per_file, 1.0f);
    return {from, to};
}

void ReadOutputProgressStats::report_stats(progress_clock::time_point interval_end) const {
    using namespace std::chrono;
    using Seconds = duration<float>;
    ReportInfo info{};
    info.time_elapsed = duration_cast<Seconds>(interval_end - m_monitoring_start_time).count();
    info.interval_time_elapsed = duration_cast<Seconds>(interval_end - m_interval_start).count();

    if (m_post_processing_stats) {
        info.interval_reads_processed = m_post_processing_stats->interval_reads_processed;
        info.total_reads_processed = m_post_processing_stats->total_reads_processed;
        info.total_reads_estimate = m_post_processing_stats->total_reads_estimate;
    } else {
        info.interval_reads_processed = m_interval_previous_stat_collectors_total +
                                        m_current_reads_written_count - m_interval_start_count;

        // Total number read from file may be higher than total number written by HtsWriter
        // if input files contained duplicate read_ids. Use the total number written so that
        // the sum of the intervals matches the total number processed.
        info.total_reads_processed =
                m_previous_stat_collectors_total + m_current_reads_written_count;

        if (is_known_total_number_input_reads()) {
            info.total_reads_estimate = m_total_known_readcount;
        } else {
            info.total_reads_estimate =
                    get_adjusted_estimated_total_reads(m_current_reads_written_count);
        }
    }

    if (info.total_reads_processed > 0) {
        auto estimated_total_time =
                info.time_elapsed *
                (static_cast<float>(info.total_reads_estimate) / info.total_reads_processed);
        info.time_remaining = estimated_total_time - info.time_elapsed;

        float progress = std::min(100.f, 100.f * static_cast<float>(info.total_reads_processed) /
                                                 info.total_reads_estimate);
        progress = (1 - m_post_processing_percentage) * progress;
        auto post_processing_progress = m_post_processing_percentage * m_post_processing_progress;
        progress += post_processing_progress;

        auto [from, to] = get_current_range();
        progress = from * 100.0f + (to - from) * progress;

        info.estimated_percentage = static_cast<std::size_t>(progress);
    } else {
        info.time_remaining = 0.0;
        info.estimated_percentage = 0;
    }

    show_report(info);
}

void ReadOutputProgressStats::update_stats(const stats::NamedStats& stats) {
    if (is_disabled()) {
        return;
    }
    std::lock_guard lock(m_mutex);
    m_current_reads_written_count = get_num_reads_written(stats);
}

void ReadOutputProgressStats::notify_stats_collector_completed(const stats::NamedStats& stats) {
    if (is_disabled()) {
        return;
    }

    assert(m_num_input_files > 0);

    std::lock_guard lock(m_mutex);

    const auto stats_reads_written = get_num_reads_written(stats);

    // Use m_total_known_readcount instead of calculating from the given stats
    // as m_total_known_readcount will include any duplicate ids missing from the stats.
    m_previous_stat_collectors_total += stats_reads_written;
    assert(m_previous_stat_collectors_total <= m_total_known_readcount &&
           "Expected that update_reads_per_file_estimate called before "
           "notify_stats_collector_completed");
    // HtsWriter stats don't include any duplicate read_ids, but HtsReader will not filter out duplicate read_ids
    // so account for any discrepancy in this interval
    auto duplicates_read_ids_this_interval =
            static_cast<std::size_t>(m_total_known_readcount - m_previous_stat_collectors_total);
    if (duplicates_read_ids_this_interval > 0) {
        m_previous_stat_collectors_total += duplicates_read_ids_this_interval;
    }

    m_interval_previous_stat_collectors_total +=
            stats_reads_written - m_interval_start_count + duplicates_read_ids_this_interval;
    m_interval_start_count = 0;

    m_current_reads_written_count = 0;

    if (m_stats_collection_mode == StatsCollectionMode::collector_per_input_file) {
        // entering post processing so capture read stats
        m_post_processing_stats = StatsForPostProcessing{};  // make_optional not working with clang
        m_post_processing_stats->interval_reads_processed =
                m_interval_previous_stat_collectors_total;
        m_post_processing_stats->total_reads_processed = m_previous_stat_collectors_total;
        m_post_processing_stats->total_reads_estimate = calc_total_reads_collector_per_file(0);
    }
}

std::size_t ReadOutputProgressStats::calc_total_reads_single_collector(
        std::size_t current_reads_count) const {
    // Check if we've processed more reads than we have been informed about
    // If so the excess reads must belong to the current input file being
    // processed by an HtsReader.
    std::size_t current_input_file_reads = current_reads_count > m_total_known_readcount
                                                   ? current_reads_count - m_total_known_readcount
                                                   : 0;

    spdlog::trace("num reads known: {} | processed: {} | current file: {} | estimate per file {}",
                  m_total_known_readcount, current_reads_count, current_input_file_reads,
                  m_estimated_num_reads_per_file);
    // If the reads processed from the current file is less than our expected reads
    // per file we can simply return an estimate based on this expectation
    // per file.
    if (static_cast<float>(current_input_file_reads) <= m_estimated_num_reads_per_file) {
        return to_size_t(m_estimated_num_reads_per_file * m_num_input_files);
    }

    // We must adjust our estimated reads per file upwards by including the current file in
    // the calculation.
    const auto expected_reads_current_file =
            to_size_t(current_input_file_reads / ASSUMED_PERCENTAGE_THROUGH_INPUT_FILE);
    const auto count_including_current_file = m_total_known_readcount + expected_reads_current_file;
    const auto num_known_files_including_current = m_num_files_where_readcount_known + 1;
    const auto adjusted_estimated_reads_per_file =
            static_cast<float>(count_including_current_file) / num_known_files_including_current;
    spdlog::trace("ADJUSTED num reads known: {} | current file: {} | estimate per file {}",
                  count_including_current_file, current_input_file_reads,
                  adjusted_estimated_reads_per_file);
    return to_size_t(adjusted_estimated_reads_per_file * m_num_input_files);
}

std::size_t ReadOutputProgressStats::calc_total_reads_collector_per_file(
        std::size_t current_reads_count) const {
    if (static_cast<float>(current_reads_count) <= m_estimated_num_reads_per_file) {
        return to_size_t(m_estimated_num_reads_per_file * m_num_input_files);
    }

    // Current file exceed reads per file estimate so recalculate including an
    // estimate of curremnt input file size
    auto assumed_current_file_total_reads =
            current_reads_count / ASSUMED_PERCENTAGE_THROUGH_INPUT_FILE;
    auto total_reads_including_current = m_total_known_readcount + assumed_current_file_total_reads;
    float adjusted_estimated_reads_per_file =
            total_reads_including_current /
            static_cast<float>(m_num_files_where_readcount_known + 1);

    return to_size_t(adjusted_estimated_reads_per_file * m_num_input_files);
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
    assert(!is_known_total_number_input_reads() && "More files updates supplied than input files.");
    ++m_num_files_where_readcount_known;
    m_total_known_readcount += num_reads_in_file;
    m_estimated_num_reads_per_file =
            static_cast<float>(m_total_known_readcount) / m_num_files_where_readcount_known;

    spdlog::trace("num known files: {}, reads input: {}, estimate per file {}",
                  m_num_files_where_readcount_known, m_total_known_readcount,
                  m_estimated_num_reads_per_file);
}

void ReadOutputProgressStats::notify_post_processing_completed() {
    if (m_stats_collection_mode == StatsCollectionMode::single_collector) {
        return;
    }
    std::lock_guard lock(m_mutex);
    assert(m_num_completed_files < m_num_input_files);
    ++m_num_completed_files;
    m_post_processing_stats = std::nullopt;
    m_post_processing_progress = 0.0f;
}

bool ReadOutputProgressStats::is_known_total_number_input_reads() const {
    return m_num_files_where_readcount_known == m_num_input_files;
}

bool ReadOutputProgressStats::is_disabled() const {
    return m_interval_duration == std::chrono::seconds{0};
}

void ReadOutputProgressStats::set_post_processing_percentage(float post_processing_percentage) {
    assert(post_processing_percentage <= 1.0f && post_processing_percentage >= 0.0f);
    m_post_processing_percentage = post_processing_percentage;
}

void ReadOutputProgressStats::update_post_processing_progress(float progress) {
    std::lock_guard lock(m_mutex);
    m_post_processing_progress = progress;
}

}  // namespace dorado
