#pragma once

#include <chrono>
#include <cstdint>

namespace dorado {
namespace timer {

/// \brief  Minimal timer object to facilitate recording time spans using the high resolution clock.
///         Starts a clock when constructed.
class TimerHighRes {
public:
    TimerHighRes() : m_start_time(get_current_time()) {}

    void Reset() { *this = {}; }

    int64_t GetElapsedSeconds() const {
        std::chrono::time_point<std::chrono::high_resolution_clock> now{get_current_time()};
        return std::chrono::duration_cast<std::chrono::seconds>(now - m_start_time).count();
    }

    int64_t GetElapsedMilliseconds() const {
        std::chrono::time_point<std::chrono::high_resolution_clock> now{get_current_time()};
        return std::chrono::duration_cast<std::chrono::milliseconds>(now - m_start_time).count();
    }

    int64_t GetElapsedMicroseconds() const {
        std::chrono::time_point<std::chrono::high_resolution_clock> now{get_current_time()};
        return std::chrono::duration_cast<std::chrono::microseconds>(now - m_start_time).count();
    }

    int64_t GetElapsedNanoseconds() const {
        std::chrono::time_point<std::chrono::high_resolution_clock> now{get_current_time()};
        return std::chrono::duration_cast<std::chrono::nanoseconds>(now - m_start_time).count();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start_time;

    std::chrono::time_point<std::chrono::high_resolution_clock> get_current_time() const {
        return std::chrono::high_resolution_clock::now();
    }
};

}  // namespace timer
}  // namespace dorado