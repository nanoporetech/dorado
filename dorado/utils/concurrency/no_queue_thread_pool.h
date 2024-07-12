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
#include <optional>
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
//
// N.B. If the pool is initialised with N threads then the actual number of
// threads managed by the pool will be 2*N, allowing the pool to expand under
// certain circumstances.
//
// This is to support the usecase where N threads are under full load running
// normal priority tasks and then high priority tasks start to be received.
// Instead of blocking these high prio tasks the pool is allowed to expand up to 2*N.
//
// There is a similar behaviour for normal tasks but only N/4 expansion threads are
// allowed, this is to ensure normal pipelines aren't "frozen" while high prioity
// tasks are being processed.
class NoQueueThreadPool {
public:
    NoQueueThreadPool(std::size_t num_threads);
    NoQueueThreadPool(std::size_t num_threads, std::string name);
    ~NoQueueThreadPool();

    void send(TaskType task, detail::PriorityTaskQueue::TaskQueue& task_queue);

    void join();

    class ThreadPoolQueue {
    public:
        virtual void push(TaskType task) = 0;
    };
    std::unique_ptr<ThreadPoolQueue> create_task_queue(TaskPriority priority);

private:
    std::string m_name{"async_task_exec"};
    const std::size_t m_num_threads;
    const std::size_t m_num_expansion_low_prio_threads;
    std::vector<std::thread> m_threads{};
    std::atomic_bool m_done{false};  // Note that this flag is only accessed by the managed threads.

    std::mutex m_mutex{};
    detail::PriorityTaskQueue m_priority_task_queue{};
    std::condition_variable m_message_received{};
    std::size_t m_normal_prio_tasks_in_flight{};
    std::size_t m_high_prio_tasks_in_flight{};

    void decrement_in_flight_tasks(TaskPriority priority);
    void initialise();
    detail::WaitingTask decrement_in_flight_and_wait_on_next_task(
            std::optional<TaskPriority> last_task_priority);
    void process_task_queue();
    bool try_pop_next_task(detail::WaitingTask& next_task);
    std::size_t num_tasks_in_flight();

    class ThreadPoolQueueImpl : public ThreadPoolQueue {
        NoQueueThreadPool* m_parent;
        std::unique_ptr<detail::PriorityTaskQueue::TaskQueue> m_task_queue;

    public:
        ThreadPoolQueueImpl(NoQueueThreadPool* parent,
                            std::unique_ptr<detail::PriorityTaskQueue::TaskQueue> task_queue);
        void push(TaskType task) override;
    };
};

}  // namespace dorado::utils::concurrency
