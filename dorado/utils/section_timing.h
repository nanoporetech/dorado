#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <string>

namespace dorado::utils::timings {

/**
 * report any timings that may have been instrumented
 */
void report();

namespace details {

void add_report_provider(const std::string &section_name,
                         std::function<std::string()> report_provider);

}  // namespace details

template <typename T>
class SectionTiming {
private:
    using clock = std::chrono::high_resolution_clock;
    static std::atomic<int64_t> duration_ns;
    static std::atomic<std::size_t> count;

    const clock::time_point m_enter;
    static std::once_flag register_once;

public:
    SectionTiming(std::string name) : m_enter(clock::now()) {
        std::call_once(register_once, [&name] {
            details::add_report_provider("SectionTiming_" + name, []() { return report(); });
        });
        count.fetch_add(1, std::memory_order_relaxed);
    }

    ~SectionTiming() {
        duration_ns.fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - m_enter)
                        .count(),
                std::memory_order_relaxed);
    }

    static std::string report() {
        std::chrono::nanoseconds total_duration{duration_ns};
        return "count: " + std::to_string(count.load()) + ", duration (ms): " +
               std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(total_duration)
                                      .count());
    }
};

template <typename T>
std::atomic<int64_t> SectionTiming<T>::duration_ns{};

template <typename T>
std::atomic<std::size_t> SectionTiming<T>::count{};

template <typename T>
std::once_flag SectionTiming<T>::register_once{};

}  // namespace dorado::utils::timings

/**
 * DORADO_SECTION_TIMING macro
 * Usage: add DORADO_SECTION_TIMING(); as the first line to a function requiring profiling, e.g.
 * class MyClass {
 * void some_function() {
 *     DORADO_SECTION_TIMING("MyClass::some_function");
 *     ...
 * }
 * };
 *
 * To enable ensure DORADO_SECTION_TIMING_ENABLED is defined
 * and any instrumented functions will be reported to stdout assuming that, on exit, the application calls:
 * dorado::utils::timings::report();
 *
 * output:
 * SectionTiming_MyClass::some_function : count: 1787519, duration (ms): 152544
 */
#define DORADO_SECTION_TIMING_ENABLED  // comment/uncomment to enable
#ifdef DORADO_SECTION_TIMING_ENABLED
#define DORADO_SECTION_TIMING(name)                                                       \
    class dorado_section_timing_t final                                                   \
            : private dorado::utils::timings::SectionTiming<dorado_section_timing_t> {    \
    public:                                                                               \
        dorado_section_timing_t()                                                         \
                : dorado::utils::timings::SectionTiming<dorado_section_timing_t>(name) {} \
    } dorado_section_timing
#else
#define DORADO_SECTION_TIMING(name)
#endif
