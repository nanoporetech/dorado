#include "no_queue_thread_pool.h"

#include "utils/thread_naming.h"

namespace dorado::utils::concurrency {

NoQueueThreadPool::NoQueueThreadPool(std::size_t num_threads) : m_num_threads{num_threads} {
    initialise();
}

NoQueueThreadPool::NoQueueThreadPool(std::size_t num_threads, std::string name)
        : m_name{name}, m_num_threads{num_threads} {
    initialise();
}

void NoQueueThreadPool::initialise() {
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
            m_task_queue.emplace(std::make_shared<WaitingTask>(
                    [this] { m_done.store(true, std::memory_order_relaxed); }));
        }
        m_message_received.notify_one();
    }
    for (auto & worker : m_threads) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void NoQueueThreadPool::send(TaskType task) {
    auto waiting_task = std::make_shared<WaitingTask>(std::move(task));
    {
        std::unique_lock lock(m_mutex);
        m_task_queue.push(waiting_task);
    }
    m_message_received.notify_one();
    waiting_task->started.wait();
}

std::shared_ptr<NoQueueThreadPool::WaitingTask> NoQueueThreadPool::wait_on_next_task() {
    std::unique_lock lock(m_mutex);
    m_message_received.wait(lock, [this] { return !m_task_queue.empty(); });
    auto result = std::move(m_task_queue.front());
    m_task_queue.pop();
    return result;
}

void NoQueueThreadPool::process_task_queue() {
    set_thread_name(m_name);
    while (!m_done.load(std::memory_order_relaxed)) {
        auto waiting_task = wait_on_next_task();
        waiting_task->started.signal();
        waiting_task->task();
    }
}

}  // namespace dorado::utils::concurrency