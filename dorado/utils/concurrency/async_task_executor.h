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
#include <type_traits>
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
class NoQueueThreadPool {
public:
    using TaskType = std::function<void()>;

    NoQueueThreadPool(std::size_t num_threads);
    NoQueueThreadPool(std::size_t num_threads, std::string name);
    ~NoQueueThreadPool();

    template <typename T,
              typename std::enable_if<std::is_copy_constructible<T>{}, bool>::type = true>
    void send(T&& task) {
        send_impl(task);
    }

    template <typename T,
              typename std::enable_if<!std::is_copy_constructible<T>{}, bool>::type = true>
    void send(T&& task) {
        // The task contains a non-copyable such as a SimplexReadPtr so wrap it in a
        // shared_ptr so it can be assigned to a std::function
        send_impl([task_wrapper = std::make_shared<std::decay_t<T>>(std::forward<T>(
                           task))]() -> decltype(auto) { return (*task_wrapper)(); });
    }

    void join();

private:
    struct WaitingTask {
        WaitingTask(TaskType task_) : task(std::move(task_)) {}
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

    void send_impl(TaskType task);
    void initialise();
    std::shared_ptr<WaitingTask> wait_on_next_task();
    void process_task_queue();
};

}  // namespace dorado::utils::concurrency