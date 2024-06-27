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

public:
    AsyncTaskExecutor(std::shared_ptr<NoQueueThreadPool> thread_pool);

    template <typename T>
    void send(T&& task) {
        m_thread_pool->send(std::forward<T>(task));
    }
};

}  // namespace dorado::utils::concurrency