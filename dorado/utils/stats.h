#pragma once

#include <atomic>
#include <chrono>
#include <fstream>
#include <functional>
#include <optional>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace dorado {
namespace stats {

using NamedStats = std::unordered_map<std::string, double>;
using StatsReporter = std::function<std::tuple<std::string, NamedStats>()>;
using StatsCallable = std::function<void(const NamedStats&)>;

class StatsSampler {
public:
    // Takes 2 arguments
    // - a vector of callable objects that will be periodically queried
    //   to retrive stats. State to which they refer must outlive this object.
    // - a vector of callable objects that are given the queried stats at
    //   the same period. Useful for analysis or post processing of stats.
    StatsSampler(std::chrono::system_clock::duration sampling_period,
                 std::vector<StatsReporter> stats_reporters,
                 std::vector<StatsCallable> stats_callables)
            : m_stats_reporters(std::move(stats_reporters)),
              m_stats_callables(std::move(stats_callables)),
              m_sampling_period(sampling_period),
              m_sampling_thread(&StatsSampler::sampling_thread_fn, this) {}

    ~StatsSampler() {
        if (m_sampling_thread.joinable())
            terminate();
    }

    void terminate() {
        m_should_terminate = true;
        m_sampling_thread.join();
    }

    // Dumps stats in CSV form, with entries filtered optionally according to name_filter.
    void dump_stats(std::ofstream& out_stream,
                    std::optional<std::regex> name_filter = std::nullopt) const {
        if (m_records.empty())
            return;

        // Determine the set of stats names across all time steps, filtered according
        // to the user-specified criterion.
        // Use a std::set so we get an ordering sorted by name.
        std::set<std::string> stat_names;
        for (const auto& [elapsed_ms, record] : m_records) {
            for (const auto& [name, value] : record) {
                if (!name_filter.has_value() || std::regex_match(name, name_filter.value()))
                    stat_names.insert(name);
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

private:
    std::vector<StatsReporter> m_stats_reporters;  // Entities we monitor
    std::vector<StatsCallable> m_stats_callables;
    std::atomic<bool> m_should_terminate{false};
    std::chrono::system_clock::duration m_sampling_period;
    std::chrono::time_point<std::chrono::system_clock> m_start_time;
    std::thread m_sampling_thread;

    // Stats returned by nodes, and recorded per sample time,
    // are of this form.
    struct StatsRecord {
        int64_t elapsed_ms;
        NamedStats stats;
    };
    std::vector<StatsRecord> m_records;

    void sampling_thread_fn() {
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
                    std::chrono::duration_cast<std::chrono::milliseconds>(now - m_start_time)
                            .count();

            // Sample stats from each node.
            for (const auto& reporter : m_stats_reporters) {
                const auto [obj_name, obj_stats] = reporter();
                for (const auto& [name, value] : obj_stats)
                    stats_record.stats[obj_name + "." + name] = value;
            }

            m_records.push_back(stats_record);

            for (auto& c : m_stats_callables) {
                c(stats_record.stats);
            }
        }
    }
};

// Constructs a callable StatsReporter object based on an object
// that implements get_name / sample_stats.
template <class T>
StatsReporter make_stats_reporter(const T& node) {
    return [&node]() { return std::make_tuple(node.get_name(), node.sample_stats()); };
}

// Returns NamedStats containing the given object's stats. with their names prefixed by
// the object's name.
template <class T>
NamedStats from_obj(const T& obj) {
    NamedStats prefixed_stats;
    const auto prefix = obj.get_name();
    const auto obj_stats = obj.sample_stats();
    for (const auto& [name, value] : obj_stats) {
        prefixed_stats[prefix + "." + name] = value;
    }
    return prefixed_stats;
}

// Minimal timer object to facilitate recording time spans.
// Starts a clock when constructed which can be queried in ms subsequently.
class Timer {
public:
    Timer() : m_start_time(std::chrono::system_clock::now()) {}
    int64_t GetElapsedMS() {
        auto elapsed_time = std::chrono::system_clock::now() - m_start_time;
        return std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count();
    }

private:
    std::chrono::time_point<std::chrono::system_clock> m_start_time;
};

}  // namespace stats
}  // namespace dorado
