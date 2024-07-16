#include "priority_task_queue.h"

#include <cassert>
#include <iterator>

namespace dorado::utils::concurrency::detail {

class PriorityTaskQueue::ProducerQueue : public TaskQueue {
    PriorityTaskQueue* m_parent;
    TaskPriority m_priority;
    std::queue<TaskType> m_producer_queue{};

public:
    ProducerQueue(PriorityTaskQueue* parent, TaskPriority priority);

    TaskPriority priority() const { return m_priority; };

    void push(TaskType task) override;  // queue.push(task), if size==1 parent.push
    TaskType pop();                     // pop, if not empty parent.push
};

//void PriorityTaskQueue::push(std::shared_ptr<WaitingTask> task) {
//    m_task_list.push_back(std::move(task));
//    auto task_itr = std::prev(m_task_list.end());
//    if ((*task_itr)->priority == TaskPriority::high) {
//        m_high_queue.push(task_itr);
//    } else {
//        m_low_queue.push(task_itr);
//    }
//}

void PriorityTaskQueue::queue_producer_task(ProducerQueue* producer_queue) {
    m_producer_queue_list.push_back(producer_queue);
    auto task_itr = std::prev(m_producer_queue_list.end());
    if (producer_queue->priority() == TaskPriority::high) {
        m_high_producer_queue.push(task_itr);
    } else {
        m_low_producer_queue.push(task_itr);
    }
}

std::size_t PriorityTaskQueue::size() const { return m_num_high_prio + m_num_normal_prio; }

std::size_t PriorityTaskQueue::size(TaskPriority priority) const {
    return priority == TaskPriority::high ? m_num_high_prio : m_num_normal_prio;
}

//std::shared_ptr<WaitingTask> PriorityTaskQueue::old_pop() {
//    auto task_itr = m_task_list.begin();
//    if (!m_low_queue.empty() && task_itr == m_low_queue.front()) {
//        m_low_queue.pop();
//    } else {
//        m_high_queue.pop();
//    }
//    auto result = std::move(m_task_list.front());
//    m_task_list.pop_front();
//    return result;
//}
//
//std::shared_ptr<WaitingTask> PriorityTaskQueue::old_pop(TaskPriority priority) {
//    WaitingTaskList::iterator task_itr;
//    if (priority == TaskPriority::high) {
//        task_itr = m_high_queue.front();
//        m_high_queue.pop();
//    } else {
//        task_itr = m_low_queue.front();
//        m_low_queue.pop();
//    }
//    auto result = std::move(*task_itr);
//    m_task_list.erase(task_itr);
//    return result;
//}

WaitingTask PriorityTaskQueue::pop() {
    auto producer_queue_itr = m_producer_queue_list.begin();
    TaskPriority popped_priority{TaskPriority::normal};
    if (!m_low_producer_queue.empty() && producer_queue_itr == m_low_producer_queue.front()) {
        m_low_producer_queue.pop();
    } else {
        popped_priority = TaskPriority::high;
        m_high_producer_queue.pop();
    }

    WaitingTask result{(*producer_queue_itr)->pop(), popped_priority};
    m_producer_queue_list.pop_front();
    return result;
}

WaitingTask PriorityTaskQueue::pop(TaskPriority priority) {
    ProducerQueueList::iterator producer_queue_itr;
    if (priority == TaskPriority::high) {
        producer_queue_itr = m_high_producer_queue.front();
        m_high_producer_queue.pop();
    } else {
        producer_queue_itr = m_low_producer_queue.front();
        m_low_producer_queue.pop();
    }

    WaitingTask result{(*producer_queue_itr)->pop(), priority};
    m_producer_queue_list.pop_front();
    return result;
}

bool PriorityTaskQueue::empty() const { return size() == 0; }

bool PriorityTaskQueue::empty(TaskPriority priority) const { return size(priority) == 0; }

PriorityTaskQueue::ProducerQueue::ProducerQueue(PriorityTaskQueue* parent, TaskPriority priority)
        : m_parent(parent), m_priority(priority) {}

void PriorityTaskQueue::ProducerQueue::push(TaskType task) {
    m_producer_queue.push(std::move(task));
    if (m_priority == TaskPriority::normal) {
        ++m_parent->m_num_normal_prio;
    } else {
        ++m_parent->m_num_high_prio;
    }
    if (m_producer_queue.size() == 1) {
        m_parent->queue_producer_task(this);
    }
}

TaskType PriorityTaskQueue::ProducerQueue::pop() {
    assert(!m_producer_queue.empty() && "Cannot pop an empty producer queue.");
    auto result = std::move(m_producer_queue.front());
    m_producer_queue.pop();
    if (m_priority == TaskPriority::normal) {
        --m_parent->m_num_normal_prio;
    } else {
        --m_parent->m_num_high_prio;
    }
    if (!m_producer_queue.empty()) {
        m_parent->queue_producer_task(this);
    }
    return result;
}

std::unique_ptr<PriorityTaskQueue::TaskQueue> PriorityTaskQueue::create_task_queue(
        TaskPriority priority) {
    return std::make_unique<ProducerQueue>(this, priority);
}

}  // namespace dorado::utils::concurrency::detail