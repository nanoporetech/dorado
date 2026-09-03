#include "utils/concurrency/TaskPool.h"

#include <cassert>
#include <thread>

namespace dorado::utils::concurrency {

namespace {

// How long to wait for a task on "our" queue when blocking.
constexpr auto kBlockingWaitFor = std::chrono::milliseconds(100);

// How many times to try and steal work from the other queues before blocking on "our" queue.
constexpr size_t kStealLoopCount = 3;

}  // namespace

std::vector<TaskPool::TaskQueue> TaskPool::make_queues(std::size_t num_queues,
                                                       std::size_t q_capacity) {
    // All queues have the same capacity.
    // TODO: replace with std::views::repeat() when we're on C++26
    std::vector<std::size_t> sizes(num_queues, q_capacity);
    return std::vector<TaskPool::TaskQueue>{sizes.begin(), sizes.end()};
}

bool TaskPool::pop_task(Task& task, size_t worker_idx) {
    // Start at an offset so that every worker doesn't bang on the same queue.
    const std::size_t num_queues = m_task_qs.size();
    const size_t our_q_idx = worker_idx % num_queues;
    TaskQueue& our_queue = m_task_qs.at(our_q_idx);

    // Try and pop from "our" queue.
    {
        const auto status =
                our_queue.try_pop_nonblocking(task, AsyncQueueNonBlockingMode::FullLock);
        if (status == AsyncQueueStatus::Success) {
            return true;
        }
        // We never terminate the queues so this should only ever be a timeout.
        assert(status == AsyncQueueStatus::Timeout);
    }

    // Steal work if nothing is ready.
    for (std::size_t repeat = 0; repeat < kStealLoopCount; repeat++) {
        for (size_t offset = 1; offset < num_queues; offset++) {
            const size_t steal_idx = (our_q_idx + offset) % num_queues;
            auto& queue = m_task_qs.at(steal_idx);
            const auto status = queue.try_pop_nonblocking(task, AsyncQueueNonBlockingMode::TryLock);
            if (status == AsyncQueueStatus::Success) {
                return true;
            }
            // We never terminate the queues so this should only ever be a timeout.
            assert(status == AsyncQueueStatus::Timeout);
        }

        // If we couldn't find anything then yield to give the producers a chance
        // to push an item into one of the queues.
        std::this_thread::yield();
    }

    // If we didn't steal anything then wait on "our" producer for a bit.
    {
        const auto until = TaskQueue::Clock::now() + kBlockingWaitFor;
        const auto status = our_queue.try_pop_until(task, until);
        if (status == AsyncQueueStatus::Success) {
            return true;
        }
        // We never terminate the queues so this should only ever be a timeout.
        assert(status == AsyncQueueStatus::Timeout);
    }

    // No task was popped.
    return false;
}

TaskPool::TaskPool(std::size_t num_queues, std::size_t q_capacity)
        : m_task_qs(make_queues(num_queues, q_capacity)), m_q_counters(num_queues) {}

TaskPool::~TaskPool() = default;

void TaskPool::run_task(size_t worker_idx) {
    Task task;
    if (pop_task(task, worker_idx)) {
        task();
    }
}

void TaskPool::wait_for_queue_to_complete(std::size_t q_idx) {
    auto& counter = m_q_counters.at(q_idx).value;

    // Wait for the final task to signal that it's done.
    while (true) {
        const std::size_t current_count = counter.load(std::memory_order_acquire);
        if (current_count == 0) {
            break;
        }
        counter.wait(current_count, std::memory_order_acquire);
    }
}

}  // namespace dorado::utils::concurrency
