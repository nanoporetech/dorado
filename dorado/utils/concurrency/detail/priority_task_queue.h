#pragma once

#include "utils/concurrency/synchronisation.h"
#include "utils/concurrency/task_priority.h"

#include <cstddef>
#include <functional>
#include <list>
#include <memory>
#include <queue>

namespace dorado::utils::concurrency::detail {

using TaskType = std::function<void()>;

struct WaitingTask {
    WaitingTask(TaskType task_, TaskPriority priority_)
            : task(std::move(task_)), priority(priority_) {}
    TaskType task;
    Flag started{};
    TaskPriority priority;
};

/* 
 * Queue allowing tasks to be pushed and popped, also allows pop to be called
 * with a priority which will remove and return the next task with that priority
 * from the queue.
 */
class PriorityTaskQueue {
    using WaitingTaskList = std::list<std::shared_ptr<detail::WaitingTask>>;
    WaitingTaskList m_task_list{};
    std::queue<WaitingTaskList::iterator> m_low_queue{};
    std::queue<WaitingTaskList::iterator> m_high_queue{};

public:
    void push(std::shared_ptr<WaitingTask> task);

    std::shared_ptr<WaitingTask> pop();
    std::shared_ptr<WaitingTask> pop(TaskPriority priority);

    std::size_t size() const;
    std::size_t size(TaskPriority priority) const;

    bool empty() const;
    bool empty(TaskPriority priority) const;
};

}  // namespace dorado::utils::concurrency::detail