#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>

namespace dorado::utils {

// Status return by push/pop methods.
enum class AsyncQueueStatus { Success, Timeout, Terminate };

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

    // Sets item to the next element in the queue and
    // notifies a waiting thread that the queue is not full.
    // Should only be called with the mutex held via lock.
    void pop_item(std::unique_lock<std::mutex>& lock, Item& item) {
        assert(lock.owns_lock());
        assert(!m_items.empty());
        item = std::move(m_items.front());
        m_items.pop();
        ++m_num_pops;

        // Inform a waiting thread that the queue is not full.
        lock.unlock();
        m_not_full_cv.notify_one();
    }

    // Calls process_fn on the up to max_count items in the queue,
    // popping them.  Notifies all waiting threads that the queue is
    // not full.
    // Should only be called with the mutex held via lock.
    template <class ProcessFn>
    void process_items(std::unique_lock<std::mutex>& lock, ProcessFn process_fn, size_t max_count) {
        assert(lock.owns_lock());
        assert(!m_items.empty());
        size_t num_to_pop = std::min(m_items.size(), max_count);
        for (size_t i = 0; i < num_to_pop; ++i) {
            process_fn(std::move(m_items.front()));
            m_items.pop();
        }
        m_num_pops += num_to_pop;

        // Inform all waiting threads that the queue is not full, since in general
        // we have removed > 1 item and there can be > 1 thread waiting to push.
        lock.unlock();
        m_not_full_cv.notify_all();
    }

    // Waits until the queue is not empty or we are asked to terminate.
    // Returns a unique_lock holding m_mutex.
    std::unique_lock<std::mutex> wait_for_item() {
        std::unique_lock lock(m_mutex);
        m_not_empty_cv.wait(lock, [this] { return !m_items.empty() || m_terminate; });
        // Note: don't use std::move, so we have the opportunity of NRVO on lock.
        return lock;
    }

    // Same as wait_for_item, but will also time out, returning wait_status accordingly.
    template <class Clock, class Duration>
    std::tuple<std::unique_lock<std::mutex>, bool> wait_for_item_or_timeout(
            const std::chrono::time_point<Clock, Duration>& timeout_time) {
        std::unique_lock lock(m_mutex);
        bool wait_status = m_not_empty_cv.wait_until(
                lock, timeout_time, [this] { return !m_items.empty() || m_terminate; });
        return {std::move(lock), wait_status};
    }

public:
    // Attempts to push items beyond capacity will block.
    explicit AsyncQueue(size_t capacity) : m_capacity(capacity) {}

    ~AsyncQueue() {
        // Ensure CV waits terminate before destruction.
        terminate();
    }

    // Contains std::mutex and std::condition_variable, so is not copyable or movable.
    AsyncQueue(const AsyncQueue&) = delete;
    AsyncQueue(AsyncQueue&&) = delete;
    AsyncQueue& operator=(const AsyncQueue&) = delete;
    AsyncQueue& operator=(AsyncQueue&&) = delete;

    // Attempts to add an item to the queue.
    // If the queue is full, blocks until there is space or terminate() is called.
    // If space was available and the item was added, AsyncQueueStatus::Success is
    // returned.
    // If terminate() was called, the item is not added and AsyncQueueStatus::Terminate
    // is returned.
    // Items pushed must be rvalues, since we assume sole ownership.
    AsyncQueueStatus try_push(Item&& item) {
        std::unique_lock lock(m_mutex);

        // Ensure there is space for the new item, given our limit on capacity.
        m_not_full_cv.wait(lock, [this] { return m_items.size() < m_capacity || m_terminate; });

        // We hold the mutex, and either there is space in the queue, or we have been
        // asked to terminate.
        if (m_terminate) {
            return AsyncQueueStatus::Terminate;
        }

        m_items.push(std::move(item));
        ++m_num_pushes;

        // Inform a waiting thread that there is now an item available.
        lock.unlock();
        m_not_empty_cv.notify_one();

        return AsyncQueueStatus::Success;
    }

    // Obtains the next item in the queue, potentially timing out.
    // If queue is empty:
    // If timeout is reached, but we are not terminating, returns AsyncQueueStatus::Timeout.
    // If we are terminating, returns AsyncQueueStatus::Terminate;.
    // Otherwise block until an item is added.
    template <class Clock, class Duration>
    AsyncQueueStatus try_pop_until(Item& item,
                                   const std::chrono::time_point<Clock, Duration>& timeout_time) {
        auto [lock, wait_status] = wait_for_item_or_timeout(timeout_time);

        if (wait_status == false) {
            // Condition variable timed out and the predicate returned false.
            // In this case, we don't terminate and return without any output.
            return AsyncQueueStatus::Timeout;
        }

        // Termination takes effect once all items have been popped from the queue.
        if (m_terminate && m_items.empty()) {
            return AsyncQueueStatus::Terminate;
        }

        pop_item(lock, item);
        return AsyncQueueStatus::Success;
    }

    // Obtains the next item in the queue.
    // If queue is empty:
    // If we are terminating, returns AsyncQueueStatus::Terminate.
    // Otherwise block until an item is added, upon which AsyncQueueStatus::Success
    // is returned.
    AsyncQueueStatus try_pop(Item& item) {
        auto lock = wait_for_item();

        // Termination takes effect once all items have been popped from the queue.
        if (m_terminate && m_items.empty()) {
            return AsyncQueueStatus::Terminate;
        }

        pop_item(lock, item);
        return AsyncQueueStatus::Success;
    }

    // Obtains all items in the queue, up to the limit of max_count,
    // once the lock is obtained.
    // If the lock is contended this could be more efficient than repeated
    // calls to try_pop.
    // If queue is empty:
    // If we are terminating, returns AsyncQueueStatus::Terminate.
    // Otherwise block until an item is added, upon which AsyncQueueStatus::Success
    // is returned.
    template <class ProcessFn>
    AsyncQueueStatus process_and_pop_n(ProcessFn process_fn, size_t max_count) {
        auto lock = wait_for_item();

        // Termination takes effect once all items have been popped from the queue.
        if (m_terminate && m_items.empty()) {
            return AsyncQueueStatus::Terminate;
        }

        process_items(lock, process_fn, max_count);
        return AsyncQueueStatus::Success;
    }

    // Like process_and_pop_n, except it also has a timeout.  If the queue is empty
    // and we time out before an item is added, returns AsyncQueueStatus::Timeout.
    template <class ProcessFn, class Clock, class Duration>
    AsyncQueueStatus process_and_pop_n_with_timeout(
            ProcessFn process_fn,
            size_t max_count,
            const std::chrono::time_point<Clock, Duration>& timeout_time) {
        auto [lock, wait_status] = wait_for_item_or_timeout(timeout_time);

        if (wait_status == false) {
            // Condition variable timed out and the predicate returned false.
            // In this case, we don't terminate and return without any output.
            return AsyncQueueStatus::Timeout;
        }

        // Termination takes effect once all items have been popped from the queue.
        if (m_terminate && m_items.empty()) {
            return AsyncQueueStatus::Terminate;
        }

        process_items(lock, process_fn, max_count);
        return AsyncQueueStatus::Success;
    }

    // Tells the queue to terminate any CV waits.
    // Pushes will fail and return return AsyncQueueStatus::Terminate until restart is called.
    // Pops will return AsyncQueueStatus::Terminate once the queue is empty.
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

    // Resets state to active following a terminate call.
    void restart() {
        std::lock_guard lock(m_mutex);
        m_terminate = false;
    }

    // Maximum number of items the queue can contain.
    size_t capacity() const { return m_capacity; }

    // Current number of items in the queue.  Only useful for stats sampling and
    // testing.
    size_t size() const {
        std::lock_guard lock(m_mutex);
        return m_items.size();
    }

    std::string get_name() const { return "queue"; }

    std::unordered_map<std::string, double> sample_stats() const {
        std::unordered_map<std::string, double> stats;
        std::lock_guard<std::mutex> lock(m_mutex);
        stats["items"] = double(m_items.size());
        stats["pushes"] = double(m_num_pushes);
        stats["pops"] = double(m_num_pops);
        return stats;
    }
};

}  // namespace dorado::utils