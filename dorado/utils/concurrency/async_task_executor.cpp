#include "async_task_executor.h"

#include "utils/thread_naming.h"

namespace dorado::utils::concurrency {

AsyncTaskExecutor::AsyncTaskExecutor(std::shared_ptr<NoQueueThreadPool> thread_pool,
                                     TaskPriority priority)
        : m_thread_pool(std::move(thread_pool)), m_priority(priority) {}

AsyncTaskExecutor::~AsyncTaskExecutor() { flush(); }

void AsyncTaskExecutor::send_impl(TaskType task) {
    increment_tasks_in_flight();

    m_thread_pool->send(
            [task = std::move(task), this] {
                task();
                decrement_tasks_in_flight();
            },
            m_priority);
}

std::unique_ptr<std::thread> AsyncTaskExecutor::send_async(TaskType task) {
    increment_tasks_in_flight();

    auto sending_thread = std::make_unique<std::thread>([this, task = std::move(task)]() mutable {
        m_thread_pool->send(
                [task = std::move(task), this] {
                    task();
                    decrement_tasks_in_flight();
                },
                m_priority);
    });

    return sending_thread;
}

void AsyncTaskExecutor::create_flushing_counter() {
    std::lock_guard lock(m_mutex);
    if (m_flushing_counter) {
        return;
    }
    m_flushing_counter = std::make_unique<Latch>(m_num_tasks_in_flight);
}

void AsyncTaskExecutor::flush() {
    create_flushing_counter();
    m_flushing_counter->wait();
}

void AsyncTaskExecutor::increment_tasks_in_flight() {
    std::lock_guard lock(m_mutex);
    if (m_flushing_counter) {
        throw std::runtime_error(
                "AsyncTaskExecutor::send() cannot be invoked after calling flush()");
    }
    ++m_num_tasks_in_flight;
}

void AsyncTaskExecutor::decrement_tasks_in_flight() {
    std::lock_guard lock(m_mutex);
    --m_num_tasks_in_flight;
    if (!m_flushing_counter) {
        return;
    }
    m_flushing_counter->count_down();
}

}  // namespace dorado::utils::concurrency