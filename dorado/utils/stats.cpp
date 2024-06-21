#include "stats.h"

#include "thread_naming.h"

#include <ostream>
#include <set>

namespace dorado::stats {

struct StatsSampler::StatsRecord {
    int64_t elapsed_ms;
    NamedStats stats;
};

StatsSampler::StatsSampler(std::chrono::system_clock::duration sampling_period,
                           std::vector<StatsReporter> stats_reporters,
                           std::vector<StatsCallable> stats_callables,
                           size_t max_records)
        : m_stats_reporters(std::move(stats_reporters)),
          m_stats_callables(std::move(stats_callables)),
          m_max_records(max_records),
          m_sampling_period(sampling_period),
          m_sampling_thread(&StatsSampler::sampling_thread_fn, this) {}

StatsSampler::~StatsSampler() {
    if (m_sampling_thread.joinable()) {
        terminate();
    }
}

void StatsSampler::terminate() {
    m_should_terminate = true;
    m_sampling_thread.join();
}

void StatsSampler::dump_stats(std::ostream& out_stream,
                              std::optional<std::regex> name_filter) const {
    if (m_records.empty()) {
        return;
    }

    // Determine the set of stats names across all time steps, filtered according
    // to the user-specified criterion.
    // Use a std::set so we get an ordering sorted by name.
    std::set<std::string> stat_names;
    for (const auto& [elapsed_ms, record] : m_records) {
        for (const auto& [name, value] : record) {
            if (!name_filter.has_value() || std::regex_match(name, name_filter.value())) {
                stat_names.insert(name);
            }
        }
    }

    // Emit headings.
    out_stream << "elapsed_ms";
    for (const auto& stat_name : stat_names) {
        out_stream << "," << stat_name;
    }
    out_stream << "\n";

    // Emit sampled values.
    for (const auto& [elapsed_ms, record] : m_records) {
        out_stream << elapsed_ms;
        // Iterate through stats in heading order.
        for (const auto& stat_name : stat_names) {
            const auto stat_it = record.find(stat_name);
            out_stream << ",";
            if (stat_it != record.end()) {
                out_stream << stat_it->second;
            }
        }
        out_stream << "\n";
    }
}

void StatsSampler::sampling_thread_fn() {
    utils::set_thread_name("stats_sampling");
    m_start_time = std::chrono::system_clock::now();
    while (!m_should_terminate) {
        // We could attempt to adjust for clock jitter, but so far
        // it's been 1 or 2 ms per sample, so this hasn't seemed
        // worth it.
        std::this_thread::sleep_for(m_sampling_period);
        const auto now = std::chrono::system_clock::now();

        // Create a new stats record for this sample time.
        StatsRecord stats_record;
        stats_record.elapsed_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(now - m_start_time).count();

        // Sample stats from each node.
        for (const auto& reporter : m_stats_reporters) {
            const auto [obj_name, obj_stats] = reporter();
            for (const auto& [name, value] : obj_stats) {
                stats_record.stats[std::string(obj_name).append(".").append(name)] = value;
            }
        }

        // Inform all callables of these stats.
        for (auto& c : m_stats_callables) {
            c(stats_record.stats);
        }

        // Record the stats, provided we haven't exceeded our limit.
        if (m_records.size() < m_max_records) {
            m_records.push_back(std::move(stats_record));
        }
    }
}

}  // namespace dorado::stats