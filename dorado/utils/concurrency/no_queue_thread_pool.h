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

enum class TaskPriority {
    normal,
    high,
};

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

    void send(TaskType task, TaskPriority priority);

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

    void initialise();
    std::shared_ptr<WaitingTask> wait_on_next_task();
    void process_task_queue();
};

}  // namespace dorado::utils::concurrency
