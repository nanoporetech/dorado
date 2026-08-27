#pragma once

#include "multi_queue_thread_pool.h"
#include "synchronisation.h"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>

namespace dorado::utils::concurrency {

// Executor for posting tasks to an underlying thread pool
//
// N.B. The executor itself is not thread safe, it must be used
// from a single thread such that calls to construction, send,
// flush and destruction may not interleave.
//
// Many executors can be assigned to the same thread pool.
// Allows tasks posted to a single instance of the executor
// to be waited on instead of waiting on all tasks posted to
// the underlying thread pool.
// Useful if a Node sharing a thread pool across pipelines needs
// to flush it's tasks without waiting for other pipleines to
// also flush.
class AsyncTaskExecutor {
    MultiQueueThreadPool::ThreadPoolQueue& m_thread_pool_queue;
    mutable std::mutex m_mutex;
    std::condition_variable m_tasks_in_flight_cv;
    std::size_t m_num_tasks_in_flight{};
    std::unique_ptr<Latch> m_flushing_counter;
    const std::size_t m_max_tasks_in_flight;

    void decrement_tasks_in_flight();
    void increment_tasks_in_flight();
    void create_flushing_counter();

    AsyncTaskExecutor(const AsyncTaskExecutor&) = delete;
    AsyncTaskExecutor& operator=(const AsyncTaskExecutor&) = delete;
    AsyncTaskExecutor(AsyncTaskExecutor&&) = delete;
    AsyncTaskExecutor& operator=(AsyncTaskExecutor&&) = delete;

public:
    AsyncTaskExecutor(MultiQueueThreadPool& thread_pool,
                      TaskPriority priority,
                      std::size_t max_queue_size);
    ~AsyncTaskExecutor();

    void send(TaskType&& task);

    std::size_t num_tasks_in_flight() const {
        std::lock_guard lock(m_mutex);
        return m_num_tasks_in_flight;
    }

    // Blocks until all queued tasks are completed.
    // After invoking no further tasks may be enqueued by this executor until `restart` has been called.
    void flush();

    // Allow jobs to be submitted once again.
    void restart();

    // Testability. Do NOT use outside utests
    std::unique_ptr<std::thread> send_async(TaskType task);
};

}  // namespace dorado::utils::concurrency