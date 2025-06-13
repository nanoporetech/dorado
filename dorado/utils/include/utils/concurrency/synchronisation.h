#pragma once

/**
 * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 *   H E A D E R   O N L Y   I M P L E M E N T A T I O N
 * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 * 
 * Must remain a header only implementation as depended
 * upon by ont_core basecall client code which cannot 
 * have a dependency on the dorado lib.
 */

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <vector>

namespace dorado::utils::concurrency {

/**
 * use to block threads till a counter reaches zero
 */
class Latch final {
    std::size_t m_count;
    mutable std::mutex m_mutex;
    mutable std::condition_variable m_condition;

public:
    /**
     * Constructs a latch and initializes its internal counter
     *
     * @param count the initial value of the internal counter
     */
    Latch(std::size_t count) : m_count(count), m_mutex(), m_condition() {}

    ~Latch() {}

    bool is_zeroed() const {
        std::lock_guard lock(m_mutex);
        return m_count == 0;
    }

    void count_down() {
        std::lock_guard lock(m_mutex);
        bool zeroed{false};
        if (m_count > 0) {
            --m_count;
            zeroed = m_count == 0;
        }
        // We must notify under the lock here so that a thread waiting on the latch can't unblock
        //  and delete the latch while we are trying to notify on the m_condition.
        if (zeroed) {
            m_condition.notify_all();
        }
    }

    void wait() const {
        std::unique_lock lock(m_mutex);
        m_condition.wait(lock, [this] { return m_count == 0; });
    }

    template <class Duration>
    bool wait_for(const Duration &timeout) const {
        std::unique_lock lock(m_mutex);
        m_condition.wait_for(lock, timeout, [this] { return m_count == 0; });
        return m_count == 0;
    }

    template <class TimePoint>
    bool wait_until(const TimePoint &timeout_time) const {
        std::unique_lock lock(m_mutex);
        m_condition.wait_until(lock, timeout_time, [this] { return m_count == 0; });
        return m_count == 0;
    }
};

/**
 * Adapter providing a single signalled state interface over a latch with a count of one
 */
class Flag final {
    Latch m_latch;

public:
    /**
     * Constructs a flag
     */
    Flag() : m_latch(1) {}

    void signal() { m_latch.count_down(); }

    bool is_signalled() const { return m_latch.is_zeroed(); }

    void wait() const { m_latch.wait(); }

    template <class Duration>
    bool wait_for(const Duration &timeout) const {
        return m_latch.wait_for(timeout);
    }

    template <class TimePoint>
    bool wait_until(const TimePoint &timeout_time) const {
        return m_latch.wait_until(timeout_time);
    }
};

/**
 * Adapter providing multiple flag slots, all of which must be signalled for wait to return.
 */
class CompositeFlag final {
    std::vector<std::unique_ptr<Flag>> m_flags;

public:
    /**
     * Constructs a composite flag and initializes its size
     *
     * @param count the number of internal flags
     */
    explicit CompositeFlag(std::size_t size) : m_flags(size) {
        for (auto &flag : m_flags) {
            flag = std::make_unique<Flag>();
        }
    }

    Flag &operator[](std::size_t slot) { return *m_flags[slot]; }

    /// true iff all flags are signalled.
    bool is_signalled() const {
        return std::all_of(m_flags.cbegin(), m_flags.cend(),
                           [](const std::unique_ptr<Flag> &flag) { return flag->is_signalled(); });
    }

    void wait() const {
        for (auto &flag : m_flags) {
            flag->wait();
        }
    }

    template <class Duration>
    bool wait_for(const Duration &timeout) const {
        return wait_until(std::chrono::steady_clock::now() + timeout);
    }

    template <class TimePoint>
    bool wait_until(const TimePoint &timeout_time) const {
        return std::all_of(m_flags.cbegin(), m_flags.cend(),
                           [&timeout_time](const std::unique_ptr<Flag> &flag) {
                               return flag->wait_until(timeout_time);
                           });
    }
};

}  // namespace dorado::utils::concurrency