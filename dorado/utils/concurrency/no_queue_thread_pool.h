#pragma once

#include "detail/priority_task_queue.h"
#include "synchronisation.h"
#include "task_priority.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace dorado::utils::concurrency {

using TaskType = detail::TaskType;

// Thread pool which blocks new tasks being added while all
// threads are busy.
// Suitable for usecases where a producer thread should not be allowed to
// enqueue a large amount of tasks ahead of a second producer thread
// beginning to enqueue tasks.
class NoQueueThreadPool {
public:
    NoQueueThreadPool(std::size_t num_threads);
    NoQueueThreadPool(std::size_t num_threads, std::string name);
    ~NoQueueThreadPool();

    void send(TaskType task, TaskPriority priority);

    void join();

private:
    std::string m_name{"async_task_exec"};
    const std::size_t m_num_threads;
    const std::size_t m_num_expansion_low_prio_threads;
    std::vector<std::thread> m_threads{};
    std::atomic_bool m_done{false};  // Note that this flag is only accessed by the managed threads.

    std::mutex m_mutex{};
    detail::PriorityTaskQueue m_task_queue{};
    std::condition_variable m_message_received{};
    std::size_t m_normal_prio_tasks_in_flight{};
    std::size_t m_high_prio_tasks_in_flight{};

    void initialise();
    std::shared_ptr<detail::WaitingTask> wait_on_next_task(
            std::shared_ptr<detail::WaitingTask>& last_task);
    void process_task_queue();
    bool try_pop_next_task(std::shared_ptr<detail::WaitingTask>& next_task);
    std::size_t num_tasks_in_flight();
};

}  // namespace dorado::utils::concurrency
