#pragma once

#include "no_queue_thread_pool.h"
#include "synchronisation.h"

#include <memory>
#include <mutex>
#include <thread>

namespace dorado::utils::concurrency {

// Executor for posting tasks to an underlying thread pool
// Many executors can be assigned to the same thread pool.
// Allows tasks posted to a single instance of the executor
// to be waited on instead of waiting on all tasks posted to
// the underlying thread pool.
// Useful if a Node sharing a thread pool across pipelines needs
// to flush it's tasks without waiting for other pipleines to
// also flush.
class AsyncTaskExecutor {
    NoQueueThreadPool& m_thread_pool;
    const TaskPriority m_priority;
    std::mutex m_mutex{};
    std::size_t m_num_tasks_in_flight{};
    std::unique_ptr<Latch> m_flushing_counter;

    void send_impl(TaskType task);
    void decrement_tasks_in_flight();
    void increment_tasks_in_flight();
    void create_flushing_counter();

public:
    AsyncTaskExecutor(NoQueueThreadPool& thread_pool, TaskPriority priority);
    ~AsyncTaskExecutor();

    template <typename T,
              typename std::enable_if<std::is_copy_constructible<T>{}, bool>::type = true>
    void send(T&& task) {
        send_impl(std::forward<T>(task));
    }

    template <typename T,
              typename std::enable_if<!std::is_copy_constructible<T>{}, bool>::type = true>
    void send(T&& task) {
        // The task contains a non-copyable such as a SimplexReadPtr so wrap it in a
        // shared_ptr so it can be assigned to a std::function
        send_impl([task_wrapper = std::make_shared<std::decay_t<T>>(std::forward<T>(
                           task))]() -> decltype(auto) { return (*task_wrapper)(); });
    }

    // Blocks until all queued tasks are completed.
    // After invoking no further tasks may be enqueued by this executor.
    void flush();

    // Testability. Do NOT use outside utests
    std::unique_ptr<std::thread> send_async(TaskType task);
};

}  // namespace dorado::utils::concurrency