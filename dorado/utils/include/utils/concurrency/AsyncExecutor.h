#pragma once

#include "utils/concurrency/TaskPool.h"

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace dorado::utils::concurrency {

// Executor that you can push tasks to.
class AsyncExecutor {
    TaskPool* m_tasks = nullptr;
    std::size_t m_q_idx = 0;

private:
    AsyncExecutor(const AsyncExecutor&) = delete;
    AsyncExecutor& operator=(const AsyncExecutor&) = delete;

public:
    explicit AsyncExecutor(TaskPool& tasks, std::size_t q_idx) : m_tasks(&tasks), m_q_idx(q_idx) {
        if (q_idx >= m_tasks->num_queues()) {
            throw std::logic_error("Invalid queue index for pool");
        }
    }
    ~AsyncExecutor() = default;

    explicit AsyncExecutor() noexcept = default;
    AsyncExecutor(AsyncExecutor&& o) noexcept : AsyncExecutor() {
        std::swap(m_tasks, o.m_tasks);
        std::swap(m_q_idx, o.m_q_idx);
    }
    AsyncExecutor& operator=(AsyncExecutor&&) = delete;

    // Push a new task to the pool.
    template <typename Func>
    void send(Func&& func) {
        assert(m_tasks != nullptr);
        m_tasks->send(std::forward<Func>(func), m_q_idx);
    }

    // Wait for all existing tasks to finish.
    void flush();

    // How many tasks are yet to be completed.
    std::size_t tasks_in_flight() const;
};

}  // namespace dorado::utils::concurrency
