#include "async_task_executor.h"

#include "utils/thread_naming.h"

namespace dorado::utils::concurrency {

AsyncTaskExecutor::AsyncTaskExecutor(std::shared_ptr<NoQueueThreadPool> thread_pool)
        : m_thread_pool(std::move(thread_pool)) {}

void AsyncTaskExecutor::send_impl(NoQueueThreadPool::TaskType task) {
    m_thread_pool->send(std::move(task));
}

}  // namespace dorado::utils::concurrency