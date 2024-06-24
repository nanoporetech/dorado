#pragma once

#include "synchronisation.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace dorado::utils::concurrency {

//namespace details {
///**
// * Helper class providing a thread safe messaging queue which can be pushed to and waited on.
// * Pushing messages while the queue is at full capacity will cause the  pushing thread to block
// * until a message has been popped and a new message can be enqueued.
// *
// * TODO: consider a fair polling mutex to prevent the same thread consistently winning the race
// * to enqueu the next task.
// */
//template <typename T>
//class BlockingMessageQueue final {
//    typedef T message_type;
//
//    BlockingMessageQueue(const MessageQueue &) = delete;
//
//    BlockingMessageQueue &operator=(BlockingMessageQueue &) = delete;
//
//    std::mutex m_mutex;
//    std::queue<message_type> m_message_queue;
//    std::condition_variable m_message_received;
//
//public:
//    MessageQueue() = default;
//
//    auto send(message_type message) -> void;
//
//    auto receive() -> message_type;
//};
//
//}  // namespace details

// Thread pool which blocks new tasks being added while all
// threads are busy.
// Suitable for usecases where a producer thread should not be allowed to
// enqueue a large amount of tasks ahead of a second producer thread
// beginning to enqueue tasks.
class AsyncTaskExecutor {
public:
    using TaskType = std::function<void()>;

    AsyncTaskExecutor(std::size_t num_threads);
    AsyncTaskExecutor(std::size_t num_threads, std::string name);
    ~AsyncTaskExecutor();

    void send(TaskType task);
    void join();

private:
    struct WaitingTask {
        WaitingTask(TaskType task_) : task(task_) {}
        TaskType task;
        Flag started{};
    };
    std::string m_name{"async_task_exec"};
    const std::size_t m_num_threads;
    std::vector<std::thread> m_threads{};
    std::atomic_bool m_done{false};  // Note that this flag is only accessed by the managed threads.

    std::mutex m_mutex{};
    std::queue<std::shared_ptr<WaitingTask>> m_task_queue{};
    std::condition_variable m_message_received{};
    bool m_terminate{false};

    void initialise();
    std::shared_ptr<WaitingTask> wait_on_next_task();
    void process_task_queue();
};
}  // namespace dorado::utils::concurrency