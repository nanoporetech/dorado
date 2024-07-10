#include "no_queue_thread_pool.h"

#include "utils/section_timing.h"
#include "utils/thread_naming.h"

#include <algorithm>
#include <cassert>
#include <iterator>

namespace dorado::utils::concurrency {

NoQueueThreadPool::NoQueueThreadPool(std::size_t num_threads)
        : m_num_threads{num_threads},
          m_num_expansion_low_prio_threads(std::max(static_cast<std::size_t>(1),
                                                    static_cast<std::size_t>(num_threads / 4))) {
    initialise();
}

NoQueueThreadPool::NoQueueThreadPool(std::size_t num_threads, std::string name)
        : m_name{std::move(name)},
          m_num_threads{num_threads},
          m_num_expansion_low_prio_threads(std::max(static_cast<std::size_t>(1),
                                                    static_cast<std::size_t>(num_threads / 4))) {
    initialise();
}

void NoQueueThreadPool::initialise() {
    // Total number of threads are m_num_threads + expansion space for the
    // same number of high prio tasks, in case all threads are busy with a
    //// normal prio task when a high prio task arrives.
    for (std::size_t i{0}; i < m_num_threads; ++i) {
        m_threads.emplace_back([this] { process_task_queue(); });
    }
}

NoQueueThreadPool::~NoQueueThreadPool() { join(); }

void NoQueueThreadPool::join() {
    // post as many done messages as there are threads to make sure all waiting threads will receive a wakeup
    for (uint32_t thread_index{0}; thread_index < m_num_threads; ++thread_index) {
        {
            std::unique_lock lock(m_mutex);
            m_task_queue.push(std::make_shared<detail::WaitingTask>(
                    [this] { m_done.store(true, std::memory_order_relaxed); },
                    TaskPriority::normal));
        }
        m_message_received.notify_one();
    }
    for (auto& worker : m_threads) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void NoQueueThreadPool::send(TaskType task, TaskPriority priority) {
    DORADO_SECTION_TIMING("NoQueueThreadPool::send");
    auto waiting_task = std::make_shared<detail::WaitingTask>(std::move(task), priority);
    {
        std::unique_lock lock(m_mutex);
        m_task_queue.push(waiting_task);
    }
    m_message_received.notify_one();
    //waiting_task->started.wait();
}

std::size_t NoQueueThreadPool::num_tasks_in_flight() {
    return m_normal_prio_tasks_in_flight + m_high_prio_tasks_in_flight;
}

bool NoQueueThreadPool::try_pop_next_task(std::shared_ptr<detail::WaitingTask>& next_task) {
    if (m_task_queue.empty()) {
        return false;
    }
    if (num_tasks_in_flight() < m_num_threads) {
        next_task = m_task_queue.pop();
        if (next_task->priority == TaskPriority::normal) {
            ++m_normal_prio_tasks_in_flight;
        } else {
            ++m_high_prio_tasks_in_flight;
        }
        return true;
    }

    if (m_high_prio_tasks_in_flight < m_num_threads && !m_task_queue.empty(TaskPriority::high)) {
        next_task = m_task_queue.pop(TaskPriority::high);
        ++m_high_prio_tasks_in_flight;
        return true;
    }

    if (m_normal_prio_tasks_in_flight < m_num_expansion_low_prio_threads &&
        !m_task_queue.empty(TaskPriority::normal)) {
        next_task = m_task_queue.pop(TaskPriority::normal);
        ++m_normal_prio_tasks_in_flight;
        return true;
    }

    return false;
}

void NoQueueThreadPool::decrement_in_flight_tasks(TaskPriority priority) {
    if (priority == TaskPriority::normal) {
        --m_normal_prio_tasks_in_flight;
    } else {
        --m_high_prio_tasks_in_flight;
    }
}

std::shared_ptr<detail::WaitingTask> NoQueueThreadPool::decrement_in_flight_and_wait_on_next_task(
        std::optional<TaskPriority> last_task_priority) {
    // Design flaw: doing both decrement and wait, an SRP violation - weak cohesion, because we don't
    // want to unlock between the decrement and the wait, and it is a lesser "evil" than the
    // alternative design flaw of passing a lock between functions, high coupling, as threading is
    // easier to reason about if the locking is encapsulated.
    std::unique_lock lock(m_mutex);
    if (last_task_priority) {
        decrement_in_flight_tasks(*last_task_priority);
    }
    std::shared_ptr<detail::WaitingTask> next_task{};
    m_message_received.wait(lock, [this, &next_task] { return try_pop_next_task(next_task); });
    return next_task;
}

void NoQueueThreadPool::process_task_queue() {
    set_thread_name(m_name);
    std::shared_ptr<detail::WaitingTask> waiting_task{};
    while (!m_done.load(std::memory_order_relaxed)) {
        waiting_task = decrement_in_flight_and_wait_on_next_task(
                waiting_task ? std::make_optional(waiting_task->priority) : std::nullopt);
        waiting_task->started.signal();
        waiting_task->task();
    }
    if (waiting_task) {
        std::unique_lock lock(m_mutex);
        decrement_in_flight_tasks(waiting_task->priority);
        // Signal so that the condition variable will be checked by waiting threads
        // This is necessary since it is possible join() was called before all the threads
        // in the thread pool had begun waiting on the condition variable, in which case
        // the notify_one() calls in the joining code would have had no effect.
        m_message_received.notify_one();
    }
}

}  // namespace dorado::utils::concurrency