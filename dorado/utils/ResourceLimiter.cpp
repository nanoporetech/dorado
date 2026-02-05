#include "utils/ResourceLimiter.h"

#include <cassert>

namespace dorado::utils {

ResourceLimiter::ResourceLimiter(std::size_t max_size) : m_max_size(max_size) {
    assert(m_max_size > 0);
}

void ResourceLimiter::acquire(WaiterState &waiter, int64_t size) {
    // The current approach cannot support a second acquire without
    // risking a deadlock. For example, with max=2 and 2 waiters
    // holding 1 element each, if either try to acquire it'll deadlock.
    assert(waiter.reserved == 0);

    {
        std::lock_guard lock(m_mutex);

        if (can_reserve(size)) {
            // We can make the reservation now.
            waiter.reserved = size;
            m_reserved += size;

            // Return early now that the reservation has been fulfilled.
            return;
        }

        // We need to wait for space to become available, so add our state
        // to the queue and wait for it outside of the global mutex lock.
        waiter.reserved = size;
        waiter.waiting = true;
        m_waiting.push(&waiter);
    }

    // The thread that wakes us will deal with updating the global state.
    {
        std::unique_lock lock(waiter.mutex);
        waiter.condvar.wait(lock, [&] { return !waiter.waiting; });
    }
}

void ResourceLimiter::release(WaiterState &waiter) {
    if (waiter.reserved == 0) {
        return;
    }

    std::lock_guard lock(m_mutex);

    // Remove our reservation.
    m_reserved -= waiter.reserved;
    waiter.reserved = 0;

    // Wake up any waiting threads that can now do work.
    while (!m_waiting.empty()) {
        auto *first = m_waiting.front();
        const std::size_t required = first->reserved;

        if (can_reserve(required)) {
            // Pop it from the queue since we're going to let it run now.
            m_waiting.pop();

            // Do the reservation for it here so that there's no race with other
            // threads when it wakes up.
            m_reserved += required;

            // Give it a kick.
            {
                std::lock_guard g(first->mutex);
                first->waiting = false;
            }
            first->condvar.notify_one();

        } else {
            // Nothing more to wake.
            break;
        }
    }
}

bool ResourceLimiter::can_reserve(std::size_t size) const {
    // Special case used==0, ie nothing else is running, even if it uses
    // more memory than available otherwise it'll get stuck forever.
    const std::size_t used = m_reserved;
    return used + size <= m_max_size || used == 0;
}

}  // namespace dorado::utils
