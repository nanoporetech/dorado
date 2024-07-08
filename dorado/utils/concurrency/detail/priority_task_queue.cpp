#include "priority_task_queue.h"

#include <iterator>

namespace dorado::utils::concurrency::detail {

void PriorityTaskQueue::push(std::shared_ptr<WaitingTask> task) {
    m_task_list.push_back(std::move(task));
    auto task_itr = std::prev(m_task_list.end());
    if ((*task_itr)->priority == TaskPriority::high) {
        m_high_queue.push(task_itr);
    } else {
        m_low_queue.push(task_itr);
    }
}

std::size_t PriorityTaskQueue::size() const { return m_task_list.size(); }

std::size_t PriorityTaskQueue::size(TaskPriority priority) const {
    return priority == TaskPriority::high ? m_high_queue.size() : m_low_queue.size();
}

std::shared_ptr<WaitingTask> PriorityTaskQueue::pop() {
    auto task_itr = m_task_list.begin();
    if (!m_low_queue.empty() && task_itr == m_low_queue.front()) {
        m_low_queue.pop();
    } else {
        m_high_queue.pop();
    }
    auto result = std::move(m_task_list.front());
    m_task_list.pop_front();
    return result;
}

std::shared_ptr<WaitingTask> PriorityTaskQueue::pop(TaskPriority priority) {
    WaitingTaskList::iterator task_itr;
    if (priority == TaskPriority::high) {
        task_itr = m_high_queue.front();
        m_high_queue.pop();
    } else {
        task_itr = m_low_queue.front();
        m_low_queue.pop();
    }
    auto result = std::move(*task_itr);
    m_task_list.erase(task_itr);
    return result;
}

bool PriorityTaskQueue::empty() const { return size() == 0; }

bool PriorityTaskQueue::empty(TaskPriority priority) const { return size(priority) == 0; }

}  // namespace dorado::utils::concurrency::detail