#pragma once

#include "utils/concurrency/synchronisation.h"
#include "utils/concurrency/task_priority.h"

#include <cstddef>
#include <functional>
#include <list>
#include <memory>
#include <queue>
#include <vector>

namespace dorado::utils::concurrency::detail {

using TaskType = std::function<void()>;

struct WaitingTask {
    WaitingTask() {}
    WaitingTask(TaskType task_, TaskPriority priority_)
            : task(std::move(task_)), priority(priority_) {}
    TaskType task{};
    TaskPriority priority{TaskPriority::normal};
};

// A queue of queues adapter class providing a single queue interface over multiple queues.
// Popping from one queue will cause that queue to go to the back of the queue of queues to
// be popped from.
// The interface supports popping by priority.
class PriorityTaskQueue {
public:
    class TaskQueue {
    public:
        virtual ~TaskQueue() = default;
        virtual void push(TaskType task) = 0;
    };
    TaskQueue& create_task_queue(TaskPriority priority);

    WaitingTask pop();
    WaitingTask pop(TaskPriority priority);

    std::size_t size() const;
    std::size_t size(TaskPriority priority) const;

    bool empty() const;
    bool empty(TaskPriority priority) const;

private:
    class ProducerQueue : public TaskQueue {
        PriorityTaskQueue* m_parent;
        TaskPriority m_priority;
        std::queue<TaskType> m_producer_queue{};

    public:
        ProducerQueue(PriorityTaskQueue* parent, TaskPriority priority);

        TaskPriority priority() const { return m_priority; };

        void push(TaskType task) override;
        TaskType pop();
    };
    std::vector<std::unique_ptr<ProducerQueue>>
            m_queue_repository{};  // ownership of producer queues
    using ProducerQueueList = std::list<ProducerQueue*>;
    ProducerQueueList m_producer_queue_list{};
    std::queue<ProducerQueueList::iterator> m_low_producer_queue{};
    std::queue<ProducerQueueList::iterator> m_high_producer_queue{};
    std::size_t m_num_normal_prio{};
    std::size_t m_num_high_prio{};

    using WaitingTaskList = std::list<std::shared_ptr<detail::WaitingTask>>;
    WaitingTaskList m_task_list{};
    std::queue<WaitingTaskList::iterator> m_low_queue{};
    std::queue<WaitingTaskList::iterator> m_high_queue{};

    void queue_producer_task(ProducerQueue* producer_queue);
};

}  // namespace dorado::utils::concurrency::detail