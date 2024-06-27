#pragma once

#include "no_queue_thread_pool.h"

#include <memory>

namespace dorado::utils::concurrency {

// Executor for posting tasks to an underlying thread pool
// Many executors can be assigned to the same thread pool.
// Allows tasks posted to a single instance of the executor
// to be waited on instead of waiting on all tasks posted to
// the underlying thread pool.
// Useful if a Node sharing a thread pool across pipelines needs
// to flush it's tasks without waiting for other pipleines to
// aslo flush.
class AsyncTaskExecutor {
    std::shared_ptr<NoQueueThreadPool> m_thread_pool;

    void send_impl(NoQueueThreadPool::TaskType task);

public:
    AsyncTaskExecutor(std::shared_ptr<NoQueueThreadPool> thread_pool);

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
};

}  // namespace dorado::utils::concurrency