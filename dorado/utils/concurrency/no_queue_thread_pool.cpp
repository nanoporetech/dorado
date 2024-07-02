#include "no_queue_thread_pool.h"

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
    //m_all_pool_threads_started = std::make_unique<Latch>(m_num_threads * 2);
    for (std::size_t i{0}; i < m_num_threads * 2; ++i) {
        m_threads.emplace_back([this] { process_task_queue(); });
    }
    //m_all_pool_threads_started->wait()
}

NoQueueThreadPool::~NoQueueThreadPool() { join(); }

void NoQueueThreadPool::join() {
    // post as many done messages as there are threads to make sure all waiting threads will receive a wakeup
    for (uint32_t thread_index{0}; thread_index < m_num_threads * 2; ++thread_index) {
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
    auto waiting_task = std::make_shared<detail::WaitingTask>(std::move(task), priority);
    {
        std::unique_lock lock(m_mutex);
        m_task_queue.push(waiting_task);
    }
    m_message_received.notify_one();
    waiting_task->started.wait();
}

std::size_t NoQueueThreadPool::num_tasks_in_flight() {
    return m_normal_prio_tasks_in_flight + m_high_prio_tasks_in_flight;
}

bool NoQueueThreadPool::try_pop_next_task() {
    assert(!m_next_task && "try_pop_next_task when next task already assigned");
    if (m_task_queue.empty()) {
        return false;
    }
    if (num_tasks_in_flight() < m_num_threads) {
        m_next_task = m_task_queue.pop();
        if (m_next_task->priority == TaskPriority::normal) {
            ++m_normal_prio_tasks_in_flight;
        } else {
            ++m_high_prio_tasks_in_flight;
        }
        return true;
    }

    if (m_high_prio_tasks_in_flight < m_num_threads && !m_task_queue.empty(TaskPriority::high)) {
        m_next_task = m_task_queue.pop(TaskPriority::high);
        ++m_high_prio_tasks_in_flight;
        return true;
    }

    if (m_normal_prio_tasks_in_flight < m_num_expansion_low_prio_threads &&
        !m_task_queue.empty(TaskPriority::normal)) {
        m_next_task = m_task_queue.pop(TaskPriority::normal);
        ++m_normal_prio_tasks_in_flight;
        return true;
    }

    return false;
}

std::shared_ptr<detail::WaitingTask> NoQueueThreadPool::wait_on_next_task(
        std::shared_ptr<detail::WaitingTask>& last_task) {
    std::unique_lock lock(m_mutex);
    if (last_task) {
        if (last_task->priority == TaskPriority::normal) {
            --m_normal_prio_tasks_in_flight;
        } else {
            --m_high_prio_tasks_in_flight;
        }
    }
    m_message_received.wait(lock, [this] { return try_pop_next_task(); });
    std::shared_ptr<detail::WaitingTask> result{};
    std::swap(result, m_next_task);
    return result;
}

void NoQueueThreadPool::process_task_queue() {
    set_thread_name(m_name);
    std::shared_ptr<detail::WaitingTask> waiting_task{};
    while (!m_done.load(std::memory_order_relaxed)) {
        waiting_task = wait_on_next_task(waiting_task);
        waiting_task->started.signal();
        waiting_task->task();
    }
    if (waiting_task) {
        std::unique_lock lock(m_mutex);
        if (waiting_task->priority == TaskPriority::normal) {
            --m_normal_prio_tasks_in_flight;
        } else {
            --m_high_prio_tasks_in_flight;
        }
        // Signal so that the condition variable will be checked by waiting threads
        // This is necessary since it is possible join() was called before all the threads
        // in the thread pool had begun waiting on the condition variable, in which case
        // the notify_one() calls in the joining code would have had no effect.
        m_message_received.notify_one();
    }
}

}  // namespace dorado::utils::concurrency