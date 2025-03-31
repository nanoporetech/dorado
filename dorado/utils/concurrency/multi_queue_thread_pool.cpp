#include "multi_queue_thread_pool.h"

#include "utils/thread_naming.h"

#include <algorithm>

namespace dorado::utils::concurrency {

MultiQueueThreadPool::MultiQueueThreadPool(std::size_t num_threads)
        : m_num_threads{num_threads},
          m_num_expansion_low_prio_threads(std::max(static_cast<std::size_t>(1),
                                                    static_cast<std::size_t>(num_threads / 4))) {
    initialise();
}

MultiQueueThreadPool::MultiQueueThreadPool(std::size_t num_threads, std::string name)
        : m_name{std::move(name)},
          m_num_threads{num_threads},
          m_num_expansion_low_prio_threads(std::max(static_cast<std::size_t>(1),
                                                    static_cast<std::size_t>(num_threads / 4))) {
    initialise();
}

void MultiQueueThreadPool::initialise() {
    // Total number of threads are m_num_threads + expansion space for the
    // same number of high prio tasks, in case all threads are busy with a
    //// normal prio task when a high prio task arrives.
    for (std::size_t i{0}; i < m_num_threads * 2; ++i) {
        m_threads.emplace_back([this] { process_task_queue(); });
    }
}

MultiQueueThreadPool::~MultiQueueThreadPool() { join(); }

void MultiQueueThreadPool::join() {
    // post as many done messages as there are threads to make sure all waiting threads will receive a wakeup
    detail::PriorityTaskQueue::TaskQueue* terminate_task_queue;
    {
        std::lock_guard lock(m_mutex);
        terminate_task_queue = &m_priority_task_queue.create_task_queue(TaskPriority::normal);
    }
    for (uint32_t thread_index{0}; thread_index < m_num_threads * 2; ++thread_index) {
        {
            std::lock_guard lock(m_mutex);
            terminate_task_queue->push([this] { m_done.store(true, std::memory_order_relaxed); });
        }
        m_message_received.notify_one();
    }
    for (auto& worker : m_threads) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void MultiQueueThreadPool::send(TaskType task, detail::PriorityTaskQueue::TaskQueue& task_queue) {
    {
        std::lock_guard lock(m_mutex);
        task_queue.push(std::move(task));
    }
    m_message_received.notify_one();
}

std::size_t MultiQueueThreadPool::num_tasks_in_flight() {
    return m_normal_prio_tasks_in_flight + m_high_prio_tasks_in_flight;
}

bool MultiQueueThreadPool::try_pop_next_task(detail::WaitingTask& next_task) {
    if (m_priority_task_queue.empty()) {
        return false;
    }
    if (num_tasks_in_flight() < m_num_threads) {
        next_task = m_priority_task_queue.pop();
        if (next_task.priority == TaskPriority::normal) {
            ++m_normal_prio_tasks_in_flight;
        } else {
            ++m_high_prio_tasks_in_flight;
        }
        return true;
    }

    if (m_high_prio_tasks_in_flight < m_num_threads &&
        !m_priority_task_queue.empty(TaskPriority::high)) {
        next_task = m_priority_task_queue.pop(TaskPriority::high);
        ++m_high_prio_tasks_in_flight;
        return true;
    }

    if (m_normal_prio_tasks_in_flight < m_num_expansion_low_prio_threads &&
        !m_priority_task_queue.empty(TaskPriority::normal)) {
        next_task = m_priority_task_queue.pop(TaskPriority::normal);
        ++m_normal_prio_tasks_in_flight;
        return true;
    }

    return false;
}

void MultiQueueThreadPool::decrement_in_flight_tasks(TaskPriority priority) {
    if (priority == TaskPriority::normal) {
        --m_normal_prio_tasks_in_flight;
    } else {
        --m_high_prio_tasks_in_flight;
    }
}

detail::WaitingTask MultiQueueThreadPool::decrement_in_flight_and_wait_on_next_task(
        std::optional<TaskPriority> last_task_priority) {
    // Design flaw: doing both decrement and wait, an SRP violation - weak cohesion, because we don't
    // want to unlock between the decrement and the wait, and it is a lesser "evil" than the
    // alternative design flaw of passing a lock between functions, high coupling, as threading is
    // easier to reason about if the locking is encapsulated.
    std::unique_lock lock(m_mutex);
    if (last_task_priority) {
        decrement_in_flight_tasks(*last_task_priority);
    }
    detail::WaitingTask next_task{};
    m_message_received.wait(lock, [this, &next_task] { return try_pop_next_task(next_task); });
    return next_task;
}

void MultiQueueThreadPool::process_task_queue() {
    set_thread_name(m_name.c_str());
    detail::WaitingTask waiting_task{};
    std::optional<TaskPriority> last_task_priority{std::nullopt};
    while (!m_done.load(std::memory_order_relaxed)) {
        waiting_task = decrement_in_flight_and_wait_on_next_task(last_task_priority);
        last_task_priority = waiting_task.priority;
        waiting_task.task();
    }
    if (last_task_priority) {
        std::lock_guard lock(m_mutex);
        decrement_in_flight_tasks(*last_task_priority);
        // Signal so that the condition variable will be checked by waiting threads
        // This is necessary since it is possible join() was called before all the threads
        // in the thread pool had begun waiting on the condition variable, in which case
        // the notify_one() calls in the joining code would have had no effect.
        m_message_received.notify_one();
    }
}

MultiQueueThreadPool::ThreadPoolQueue::ThreadPoolQueue(
        MultiQueueThreadPool* parent,
        detail::PriorityTaskQueue::TaskQueue& task_queue)
        : m_parent(parent), m_task_queue(task_queue) {}

void MultiQueueThreadPool::ThreadPoolQueue::push(TaskType task) {
    m_parent->send(std::move(task), m_task_queue);
}

MultiQueueThreadPool::ThreadPoolQueue& MultiQueueThreadPool::create_task_queue(
        TaskPriority priority) {
    std::lock_guard lock(m_mutex);
    auto& task_queue = m_priority_task_queue.create_task_queue(priority);
    return *m_queues.emplace_back(new ThreadPoolQueue(this, task_queue));
}

}  // namespace dorado::utils::concurrency
