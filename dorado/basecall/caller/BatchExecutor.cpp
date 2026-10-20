#include "BatchExecutor.h"

#include "utils/thread_utils.h"

#include <c10/core/InferenceMode.h>

#include <map>
#include <utility>

namespace dorado::basecall {

std::shared_ptr<DeviceQueue> DeviceQueue::get(const std::string& key) {
    if (key.empty()) {
        return std::make_shared<DeviceQueue>();
    }

    static std::mutex registry_mutex;
    static std::map<std::string, std::weak_ptr<DeviceQueue>> registry;

    std::lock_guard lock(registry_mutex);
    auto& entry = registry[key];
    auto queue = entry.lock();
    if (!queue) {
        queue = std::make_shared<DeviceQueue>();
        entry = queue;
    }
    return queue;
}

BatchExecutor::BatchExecutor(CallerModel& model)
        : m_model(model),
          m_policy(model.scheduling_policy()),
          m_input_queue(DeviceQueue::get(m_policy.shared_queue_key)) {
    start_threads();
}

BatchExecutor::~BatchExecutor() { terminate(); }

std::vector<decode::DecodedChunk> BatchExecutor::execute(std::unique_ptr<InFlightBatch> work,
                                                         int num_chunks) {
    auto task = std::make_shared<BatchTask>(this, std::move(work), num_chunks);
    {
        std::lock_guard lock(m_input_queue->mut);
        m_input_queue->queue.push_back(task);
    }
    m_input_queue->cv.notify_all();

    const bool decoded_by_pool = m_policy.decode_threads > 0;
    {
        std::unique_lock lock(task->mut);
        task->cv.wait(lock, [&] { return decoded_by_pool ? task->done : task->submitted; });
    }

    if (!decoded_by_pool) {
        decode_inline(*task);
    }
    return std::move(task->out);
}

void BatchExecutor::decode_inline(BatchTask& task) {
    const size_t units = m_model.decode_units(*task.work);
    for (size_t unit = 0; unit < units; ++unit) {
        m_model.decode(*task.work, unit, task.out);
    }
    m_model.release(*task.work);
}

void BatchExecutor::terminate() {
    {
        std::lock_guard lock(m_input_queue->mut);
        m_terminate = true;
    }
    m_input_queue->cv.notify_all();
    if (m_worker_thread.joinable()) {
        m_worker_thread.join();
    }
    // The worker thread tells the decode threads to stop once it has finished.
    for (auto& thread : m_decode_threads) {
        thread.join();
    }
    m_decode_threads.clear();
}

void BatchExecutor::restart() {
    // This can be called more than once, via multiple runners.
    std::lock_guard lock(m_input_queue->mut);
    if (std::exchange(m_terminate, false)) {
        start_threads();
    }
}

void BatchExecutor::start_threads() {
    {
        std::lock_guard lock(m_decode_lock);
        m_terminate_decode = false;
    }
    m_worker_thread = std::thread([this] { worker_thread_fn(); });
    m_decode_threads.reserve(m_policy.decode_threads);
    for (size_t i = 0; i < m_policy.decode_threads; ++i) {
        m_decode_threads.emplace_back([this] { decode_thread_fn(); });
    }
}

void BatchExecutor::worker_thread_fn() {
    utils::set_thread_name(m_model.worker_thread_name().c_str());
    at::InferenceMode inference_mode_guard;
    const auto thread_scope = m_model.worker_thread_scope();

    auto& queue = *m_input_queue;
    while (true) {
        std::shared_ptr<BatchTask> task;
        {
            std::unique_lock lock(queue.mut);
            queue.cv.wait(lock, [&] {
                return (!queue.queue.empty() && queue.queue.front()->owner == this) ||
                       (queue.queue.empty() && m_terminate);
            });

            if (queue.queue.empty()) {
                break;
            }

            task = queue.queue.front();

            if (m_policy.release_point == ReleasePoint::on_submit_start) {
                // Other callers that are ready can get going immediately.
                queue.queue.pop_front();
                lock.unlock();
                queue.cv.notify_all();
            }
        }

        m_model.submit(*task->work);

        if (m_policy.decode_threads > 0) {
            task->decode_units = m_model.decode_units(*task->work);
            if (task->decode_units == 0) {
                m_model.release(*task->work);
                {
                    std::lock_guard lock(task->mut);
                    task->done = true;
                }
                task->cv.notify_one();
            } else {
                {
                    std::lock_guard lock(m_decode_lock);
                    m_decode_queue.push_back(std::move(task));
                }
                m_decode_cv.notify_all();
            }
        } else {
            {
                std::lock_guard lock(task->mut);
                task->submitted = true;
            }
            task->cv.notify_one();
        }

        if (m_policy.release_point == ReleasePoint::on_submit_return) {
            // Callers using the same queue must not submit another batch until this one has
            // completed. This avoids relying on a mutex held across submission, which was
            // shown to wake threads in an uneven order.
            {
                std::lock_guard lock(queue.mut);
                queue.queue.pop_front();
            }
            queue.cv.notify_all();
        }
    }

    {
        std::lock_guard lock(m_decode_lock);
        m_terminate_decode = true;
    }
    m_decode_cv.notify_all();
}

void BatchExecutor::decode_thread_fn() {
    utils::set_thread_name(m_model.decode_thread_name().c_str());
    at::InferenceMode inference_mode_guard;
    while (true) {
        std::shared_ptr<BatchTask> task;
        size_t unit = 0;
        {
            std::unique_lock lock(m_decode_lock);
            m_decode_cv.wait(lock, [&] { return !m_decode_queue.empty() || m_terminate_decode; });
            if (m_decode_queue.empty()) {
                return;
            }
            task = m_decode_queue.front();
            unit = task->decode_units_started++;
            // If all units have been picked up, remove the task from the queue.
            if (unit + 1 == task->decode_units) {
                m_decode_queue.pop_front();
            }
        }

        m_model.decode(*task->work, unit, task->out);

        bool all_decoded = false;
        {
            std::lock_guard lock(task->mut);
            all_decoded = ++task->decode_units_finished == task->decode_units;
        }
        if (all_decoded) {
            m_model.release(*task->work);
            {
                std::lock_guard lock(task->mut);
                task->done = true;
            }
            task->cv.notify_one();
        }
    }
}

}  // namespace dorado::basecall
