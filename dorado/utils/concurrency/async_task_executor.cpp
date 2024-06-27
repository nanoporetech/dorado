#include "async_task_executor.h"

#include "utils/thread_naming.h"

namespace dorado::utils::concurrency {

AsyncTaskExecutor::AsyncTaskExecutor(std::shared_ptr<NoQueueThreadPool> thread_pool)
        : m_thread_pool(std::move(thread_pool)) {}

}  // namespace dorado::utils::concurrency