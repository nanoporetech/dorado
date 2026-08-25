#include "batchsize_helpers.h"

#include "SpeedEntry.h"
#include "compiled_timings.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <ranges>
#include <stdexcept>

namespace dorado::batchsize_benchmarks {

namespace {

// Fallback for platforms that don't support std::ranges::to() yet.
#if !defined(_LIBCPP_VERSION)
template <template <typename...> typename Container, std::ranges::range Range>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward): ranges use rvalues.
auto to(Range && range) {
    using T = std::ranges::range_value_t<Range>;
    return Container<T>{range.begin(), range.end()};
}
#endif

}  // namespace

int pick_best_batch_size(std::span<const SpeedEntry> speeds,
                         uint64_t memory_limit,
                         float time_penalty) {
    if (speeds.empty()) {
        throw std::logic_error("Empty span passed to batch size selection");
    } else if (!compiled_cache::is_sorted(speeds)) {
        throw std::runtime_error("Speeds must be sorted");
    }

    // Filter entries to those that fit in memory.
    const auto is_below_memory_limit = [memory_limit](const SpeedEntry & entry) {
        return entry.memory_used <= memory_limit;
    };
    const auto entries_below_memory_limit =
            to<std::vector>(speeds | std::views::filter(is_below_memory_limit));
    if (entries_below_memory_limit.empty()) {
        throw std::runtime_error(
                fmt::format("No entries remaining after applying memory_limit ({})", memory_limit));
    }

    // Grab the fastest entry.
    const auto compare_speed = [](const SpeedEntry & lhs, const SpeedEntry & rhs) {
        return lhs.basecall_speed < rhs.basecall_speed;
    };
    const auto fastest_entry = std::ranges::max_element(entries_below_memory_limit, compare_speed);
    if (fastest_entry == entries_below_memory_limit.end()) {
        // |entries_below_memory_limit| isn't empty so there should be a max element.
        throw std::logic_error("max_element() of non-empty container doesn't exist");
    }

    // Apply the time penalty.
    // TODO: the existing time_penalty code seems odd since 3 is half of 1 is half of 0.
    // TODO: would |speed * (1 - speed_penalty)| be better?
    const double threshold_speed = fastest_entry->basecall_speed / (1.0 + time_penalty);
    const auto over_threshold = [threshold_speed](const SpeedEntry & entry) {
        return entry.basecall_speed >= threshold_speed;
    };
    const auto first_over_threshold = std::find_if(
            entries_below_memory_limit.begin(), entries_below_memory_limit.end(), over_threshold);
    if (first_over_threshold == entries_below_memory_limit.end()) {
        // This should be impossible since the fastest speed should be over the threshold.
        throw std::logic_error("Error in batch size selection");
    }

    spdlog::debug("Fastest capped+limited batch size is {} @ {}", first_over_threshold->batch_size,
                  first_over_threshold->basecall_speed);
    return first_over_threshold->batch_size;
}

}  // namespace dorado::batchsize_benchmarks
