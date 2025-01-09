#include "async_task_executor.h"

namespace dorado::utils::concurrency {

AsyncTaskExecutor::AsyncTaskExecutor(MultiQueueThreadPool& thread_pool,
                                     TaskPriority priority,
                                     std::size_t max_queue_size)
        : m_thread_pool_queue(thread_pool.create_task_queue(priority)),
          m_max_tasks_in_flight(max_queue_size) {}

AsyncTaskExecutor::~AsyncTaskExecutor() { flush(); }

void AsyncTaskExecutor::send_impl(TaskType task) {
    increment_tasks_in_flight();

    m_thread_pool_queue.push([task_ = std::move(task), this] {
        task_();
        decrement_tasks_in_flight();
    });
}

std::unique_ptr<std::thread> AsyncTaskExecutor::send_async(TaskType task) {
    increment_tasks_in_flight();

    auto sending_thread = std::make_unique<std::thread>([this, t = std::move(task)]() mutable {
        m_thread_pool_queue.push([task_ = std::move(t), this] {
            task_();
            decrement_tasks_in_flight();
        });
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

void AsyncTaskExecutor::restart() { m_flushing_counter.reset(); }

void AsyncTaskExecutor::increment_tasks_in_flight() {
    std::unique_lock lock(m_mutex);
    if (m_flushing_counter) {
        throw std::runtime_error(
                "AsyncTaskExecutor::send() cannot be invoked after calling flush()");
    }
    m_tasks_in_flight_cv.wait(lock,
                              [this] { return m_num_tasks_in_flight < m_max_tasks_in_flight; });
    ++m_num_tasks_in_flight;
}

void AsyncTaskExecutor::decrement_tasks_in_flight() {
    bool flush_in_progress{false};
    {
        std::lock_guard lock(m_mutex);
        --m_num_tasks_in_flight;
        if (m_flushing_counter) {
            flush_in_progress = true;
        }

        // We must notify with the lock still held to prevent the destructor being invoked
        // and completing before this thread attempts to call notify_one on the destructed cvar.
        m_tasks_in_flight_cv.notify_one();
    }

    if (flush_in_progress) {
        // The flush operation in the destructor is waiting on the m_flushing_counter.
        // After signalling m_flushing_counter the destructor may complete, so ensure
        // no longer using member variable (e.g. the cvar) and ensure the lock_guard
        // has released the mutex before signalling.
        m_flushing_counter->count_down();
    }
}

}  // namespace dorado::utils::concurrency