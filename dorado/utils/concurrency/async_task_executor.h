#pragma once

#include "synchronisation.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace dorado::utils::concurrency {

// Thread pool which blocks new tasks being added while all
// threads are busy.
// Suitable for usecases where a producer thread should not be allowed to
// enqueue a large amount of tasks ahead of a second producer thread
// beginning to enqueue tasks.
class NoQueueThreadPool {
public:
    using TaskType = std::function<void()>;

    NoQueueThreadPool(std::size_t num_threads);
    NoQueueThreadPool(std::size_t num_threads, std::string name);
    ~NoQueueThreadPool();

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

    void join();

private:
    struct WaitingTask {
        WaitingTask(TaskType task_) : task(std::move(task_)) {}
        TaskType task;
        Flag started{};
    };
    std::string m_name{"async_task_exec"};
    const std::size_t m_num_threads;
    std::vector<std::thread> m_threads{};
    std::atomic_bool m_done{false};  // Note that this flag is only accessed by the managed threads.

    std::mutex m_mutex{};
    std::queue<std::shared_ptr<WaitingTask>> m_task_queue{};
    std::condition_variable m_message_received{};

    void send_impl(TaskType task);
    void initialise();
    std::shared_ptr<WaitingTask> wait_on_next_task();
    void process_task_queue();
};

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