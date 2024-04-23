#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <iosfwd>
#include <optional>
#include <regex>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace dorado {
namespace stats {

using NamedStats = std::unordered_map<std::string, double>;
using ReportedStats = std::tuple<std::string, NamedStats>;
using StatsReporter = std::function<ReportedStats()>;
using StatsCallable = std::function<void(const NamedStats&)>;

class StatsSampler {
public:
    // sampling_period: time between instances of querying/informing/recording of stats.
    // stats_reporters: a vector of callable objects that will be periodically queried
    // to retrieve stats.  State to which they refer must outlive this object.
    // stats_callables: a vector of callable objects that are given the queried stats at
    // the same period.  Useful for analysis or post processing of stats.
    // max_records: limits the number of sample records kept in memory that would be output
    // via dump_stats.  Can be 0.
    StatsSampler(std::chrono::system_clock::duration sampling_period,
                 std::vector<StatsReporter> stats_reporters,
                 std::vector<StatsCallable> stats_callables,
                 size_t max_records);

    ~StatsSampler();

    void terminate();

    // Dumps stats in CSV form, with entries filtered optionally according to name_filter.
    void dump_stats(std::ostream& out_stream, std::optional<std::regex> name_filter) const;

private:
    std::vector<StatsReporter> m_stats_reporters;  // Entities we monitor
    std::vector<StatsCallable> m_stats_callables;
    size_t m_max_records = static_cast<size_t>(0);
    std::atomic<bool> m_should_terminate{false};
    std::chrono::system_clock::duration m_sampling_period;
    std::chrono::time_point<std::chrono::system_clock> m_start_time;
    std::thread m_sampling_thread;

    // Stats returned by nodes, and recorded per sample time,
    // are of this form.
    struct StatsRecord;
    std::vector<StatsRecord> m_records;

    void sampling_thread_fn();
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
        prefixed_stats[std::string(prefix).append(".").append(name)] = value;
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
