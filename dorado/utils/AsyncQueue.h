#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>

// Asynchronous queue for producer/consumer use.
// Items must be movable.
template <class Item>
class AsyncQueue {
    // Guards the entire structure.  Should be held while adding/removing items,
    // or interacting with m_terminate.
    // Used for not-empty and not-full CV waits.
    mutable std::mutex m_mutex;
    // Signalled when an item has been consumed, and the queue therefore has space
    // for new items.
    mutable std::condition_variable m_not_full_cv;
    // Signalled when an item has been added, and the queue therefore is not empty.
    std::condition_variable m_not_empty_cv;
    // Holds the items.
    std::queue<Item> m_items;
    // Number of items that can be added before further additions block, pending
    // consumption of items.
    size_t m_capacity = 0;
    // If true, CV waits should terminate regardless of other state.
    // Pending attempts to push or pop items will fail.
    bool m_terminate = false;
    // Stats for monitoring queue usage.
    int64_t m_num_pushes = 0;
    int64_t m_num_pops = 0;

public:
    // Attempts to push items beyond capacity will block.
    AsyncQueue(size_t capacity) : m_capacity(capacity) {}

    ~AsyncQueue() {
        // Ensure CV waits terminate before destruction.
        terminate();
    }

    // Attempts to add an item to the queue.
    // If the queue is full, this method blocks until there is space or
    // terminate() is called.
    // If space was available and the item was added, true is returned.
    // If Terminate() was called, the item is not added and false is returned.
    // Items pushed must be rvalues, since we assume sole ownership.
    bool try_push(Item&& item) {
        std::unique_lock lock(m_mutex);

        // Ensure there is space for the new item, given our limit on capacity.
        m_not_full_cv.wait(lock, [this] { return m_items.size() < m_capacity || m_terminate; });

        // We hold the mutex, and either there is space in the queue, or we have been
        // asked to terminate.
        if (m_terminate)
            return false;
        m_items.push(std::move(item));
        ++m_num_pushes;

        // Inform a waiting thread that there is now an item available.
        lock.unlock();
        m_not_empty_cv.notify_one();

        return true;
    }

    // Obtains the next item in the queue, returning true on success.
    // If the queue is empty and timeout is reached, but we are not
    // terminating, returns true without updating the reference.
    // If the queue is empty, and we are terminating, returns false.
    // Otherwise we block if the queue is empty.
    template <class Clock, class Duration>
    bool try_pop_until(Item& item, const std::chrono::time_point<Clock, Duration>& timeout_time) {
        std::unique_lock lock(m_mutex);
        // Wait until either an item is added, a timeout is hit or we're asked to terminate.
        auto cv_status = m_not_empty_cv.wait_until(
                lock, timeout_time, [this] { return !m_items.empty() || m_terminate; });
        if (cv_status == false) {
            // Condition variable timed out and the predicate returned false.
            // In this case, we don't terminate and return without any output.
            return true;
        }

        // Termination takes effect once all items have been popped from the queue.
        if (m_terminate && m_items.empty()) {
            return false;
        }

        item = std::move(m_items.front());
        m_items.pop();
        ++m_num_pops;

        // Inform a waiting thread that the queue is not full.
        lock.unlock();
        m_not_full_cv.notify_one();

        return true;
    }

    // Obtains the next item in the queue, returning true on success.
    // If the queue is empty, and we are terminating, returns false.
    // Otherwise we block if the queue is empty.
    bool try_pop(Item& item) {
        std::unique_lock lock(m_mutex);
        // Wait until the queue is non-empty, or we're asked to terminate.
        m_not_empty_cv.wait(lock, [this] { return !m_items.empty() || m_terminate; });

        // Termination takes effect once all items have been popped from the queue.
        if (m_terminate && m_items.empty()) {
            return false;
        }

        item = std::move(m_items.front());
        m_items.pop();
        ++m_num_pops;

        // Inform a waiting thread that the queue is not full.
        lock.unlock();
        m_not_full_cv.notify_one();

        return true;
    }

    // Obtains all items in the queue once the lock is obtained.
    // Return value is false if we are terminating.
    // If the lock is contended this could be more efficient than repeated
    // calls to try_pop.
    template <class ProcessFn>
    bool process_and_pop_all(ProcessFn process_fn) {
        std::unique_lock lock(m_mutex);
        // Wait until the queue is non-empty, or we're asked to terminate.
        m_not_empty_cv.wait(lock, [this] { return !m_items.empty() || m_terminate; });

        // Termination takes effect once all items have been popped from the queue.
        if (m_terminate && m_items.empty()) {
            return false;
        }

        while (!m_items.empty()) {
            process_fn(m_items.front());
            m_items.pop();
            ++m_num_pops;
        }

        // Inform all waiting threads that the queue is not full, since in general
        // we have removed > 1 item and there can be > 1 thread waiting to push.
        lock.unlock();
        m_not_full_cv.notify_all();

        return true;
    }

    // Tells the queue to terminate any CV waits.
    void terminate() {
        {
            std::lock_guard lock(m_mutex);
            m_terminate = true;
        }

        // Signal all CV waits so they examine the termination flag and finish if
        // necessary.
        // notify_all, since in general an arbitrary number of threads can be
        // inside try_push/try_pop.
        m_not_full_cv.notify_all();
        m_not_empty_cv.notify_all();
    }

    // Return the current size of the queue.
    size_t size() {
        std::lock_guard lock(m_mutex);
        return m_items.size();
    }

    std::string get_name() const { return "queue"; }

    std::unordered_map<std::string, double> sample_stats() const {
        std::unordered_map<std::string, double> stats;
        std::lock_guard<std::mutex> lock(m_mutex);
        stats["items"] = m_items.size();
        stats["pushes"] = m_num_pushes;
        stats["pops"] = m_num_pops;
        return stats;
    }
};
