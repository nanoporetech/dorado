#pragma once

#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <queue>

namespace dorado::utils {

// Class to limit access to a resource.
// Similar to a counting semaphore but with ordering.
class ResourceLimiter {
public:
    // Note: instances of this should be thread_local or have an extended lifetime so
    // that they can't be destroyed while being signalled.
    // TODO: it should be possible to replace this with an atomic<ptrdiff_t>
    struct WaiterState {
        std::mutex mutex;
        std::condition_variable condvar;
        std::size_t reserved = 0;
        bool waiting = false;
    };

    // Construct a limiter of a set size.
    explicit ResourceLimiter(std::size_t max_size);

    // Acquire |size| elements on |waiter|, blocking until they're available.
    // Precondition: |waiter| isn't holding any elements.
    void acquire(WaiterState& waiter, int64_t size);

    // Release the elements held by |waiter|.
    void release(WaiterState& waiter);

    // RAII helper to scope a reservation.
    class ScopedReservation {
        ResourceLimiter& m_limiter;
        ResourceLimiter::WaiterState& m_waiter;

    public:
        explicit ScopedReservation(ResourceLimiter& limiter,
                                   ResourceLimiter::WaiterState& waiter,
                                   int64_t size)
                : m_limiter(limiter), m_waiter(waiter) {
            m_limiter.acquire(m_waiter, size);
        }

        ~ScopedReservation() { m_limiter.release(m_waiter); }
    };

private:
    std::mutex m_mutex;
    std::queue<WaiterState*> m_waiting;
    const std::size_t m_max_size;
    std::size_t m_reserved = 0;

    bool can_reserve(std::size_t size) const;
};

}  // namespace dorado::utils
