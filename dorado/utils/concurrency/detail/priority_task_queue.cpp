#include "priority_task_queue.h"

#include <cassert>
#include <iterator>

namespace dorado::utils::concurrency::detail {

void PriorityTaskQueue::queue_producer_task(TaskQueue* producer_queue) {
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

WaitingTask PriorityTaskQueue::pop() {
    assert(!m_producer_queue_list.empty());
    const auto next_priority = m_producer_queue_list.front()->priority();
    return pop(next_priority);
}

WaitingTask PriorityTaskQueue::pop(TaskPriority priority) {
    TaskQueueList::iterator producer_queue_itr;
    if (priority == TaskPriority::high) {
        assert(!m_high_producer_queue.empty());
        producer_queue_itr = m_high_producer_queue.front();
        m_high_producer_queue.pop();
    } else {
        assert(!m_low_producer_queue.empty());
        producer_queue_itr = m_low_producer_queue.front();
        m_low_producer_queue.pop();
    }
    assert(priority == (*producer_queue_itr)->priority());

    WaitingTask result{(*producer_queue_itr)->pop(), priority};
    m_producer_queue_list.erase(producer_queue_itr);
    return result;
}

bool PriorityTaskQueue::empty() const { return size() == 0; }

bool PriorityTaskQueue::empty(TaskPriority priority) const { return size(priority) == 0; }

PriorityTaskQueue::TaskQueue::TaskQueue(PriorityTaskQueue* parent, TaskPriority priority)
        : m_parent(parent), m_priority(priority) {}

void PriorityTaskQueue::TaskQueue::push(TaskType task) {
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

TaskType PriorityTaskQueue::TaskQueue::pop() {
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

PriorityTaskQueue::TaskQueue& PriorityTaskQueue::create_task_queue(TaskPriority priority) {
    return *m_queue_repository.emplace_back(new TaskQueue(this, priority));
}

}  // namespace dorado::utils::concurrency::detail
