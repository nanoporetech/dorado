#include "async_task_executor.h"

namespace dorado::utils::concurrency {

AsyncTaskExecutor::AsyncTaskExecutor(std::size_t num_threads) : m_num_threads(num_threads) {
    for (std::size_t i{0}; i < m_num_threads; ++i) {
        m_threads.emplace_back([this] { process_task_queue(); });
    }
}

AsyncTaskExecutor::~AsyncTaskExecutor() { join(); }

void AsyncTaskExecutor::join() {
    // post as many done messages as there are threads to make sure all waiting threads will receive a wakeup
    for (uint32_t thread_index{0}; thread_index < m_num_threads; ++thread_index) {
        send([this] { m_done.store(true, std::memory_order_relaxed); });
    }
    for (auto & worker : m_threads) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

std::shared_ptr<AsyncTaskExecutor::WaitingTask> AsyncTaskExecutor::wait_on_next_task() {
    std::unique_lock lock(m_mutex);
    m_message_received.wait(lock, [this] { return !m_task_queue.empty(); });
    auto result = std::move(m_task_queue.front());
    m_task_queue.pop();
    return result;
}

void AsyncTaskExecutor::send(TaskType task) {
    auto waiting_task = std::make_shared<WaitingTask>(std::move(task));
    {
        std::unique_lock lock(m_mutex);
        m_task_queue.push(waiting_task);
    }
    m_message_received.notify_one();
    waiting_task->started.wait();
}

void AsyncTaskExecutor::process_task_queue() {
    while (!m_done.load(std::memory_order_relaxed)) {
        auto waiting_task = wait_on_next_task();
        waiting_task->started.signal();
        waiting_task->task();
    }
}

}  // namespace dorado::utils::concurrency