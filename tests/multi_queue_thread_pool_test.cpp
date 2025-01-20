#include "utils/concurrency/multi_queue_thread_pool.h"

#include "utils/PostCondition.h"
#include "utils/concurrency/synchronisation.h"

#include <catch2/catch_test_macros.hpp>

#include <utility>

#define CUT_TAG "[dorado::utils::concurrency::MultiQueueThreadPool]"
#define DEFINE_TEST(name) CATCH_TEST_CASE(CUT_TAG " " name, CUT_TAG)
#define DEFINE_TEST_FIXTURE_METHOD(name) \
    CATCH_TEST_CASE_METHOD(NoQueueThreadPoolTestFixture, CUT_TAG " " name, CUT_TAG)
#define DEFINE_CATCH_SCENARIO_METHOD(name) \
    CATCH_SCENARIO_METHOD(NoQueueThreadPoolTestFixture, CUT_TAG " " name, CUT_TAG)

using namespace std::chrono_literals;

namespace dorado::utils::concurrency::multi_queue_thread_pool {

namespace {

constexpr auto TIMEOUT{10s};
constexpr auto FAST_TIMEOUT{100ms};  // when checking

std::vector<std::thread> create_producer_threads(MultiQueueThreadPool::ThreadPoolQueue& pool_queue,
                                                 std::size_t count,
                                                 const std::function<void()>& task) {
    std::vector<std::thread> producer_threads{};
    for (std::size_t index{0}; index < count; ++index) {
        producer_threads.emplace_back([&pool_queue, task] { pool_queue.push(task); });
    }

    return producer_threads;
}

class NoQueueThreadPoolTestFixture {
protected:
    const std::size_t MAX_TASKS{32};
    const std::size_t DEFAULT_THREAD_POOL_SIZE{2};
    std::vector<std::unique_ptr<Flag>> task_release_flags{};
    std::vector<std::unique_ptr<Flag>> task_started_flags{};
    std::unique_ptr<MultiQueueThreadPool> cut{};
    std::vector<std::unique_ptr<std::thread>> producer_threads{};

    void release_all_tasks() {
        for (auto& release_flag : task_release_flags) {
            release_flag->signal();
        }
    };

    auto create_task(std::size_t index) {
        return [this, index] {
            task_started_flags[index]->signal();
            task_release_flags[index]->wait();
        };
    };

    void initialise(std::size_t thread_pool_size) {
        cut = std::make_unique<MultiQueueThreadPool>(thread_pool_size, "test_executor");
    }

public:
    NoQueueThreadPoolTestFixture() {
        task_release_flags.reserve(MAX_TASKS);
        task_started_flags.reserve(MAX_TASKS);
        for (std::size_t index{0}; index < MAX_TASKS; ++index) {
            task_release_flags.emplace_back(std::make_unique<Flag>());
            task_started_flags.emplace_back(std::make_unique<Flag>());
        }
        initialise(DEFAULT_THREAD_POOL_SIZE);
    }

    ~NoQueueThreadPoolTestFixture() {
        release_all_tasks();
        for (auto& producer_thread : producer_threads) {
            if (producer_thread && producer_thread->joinable()) {
                producer_thread->join();
            }
        }
    }
};

}  // namespace

DEFINE_TEST("create_task_queue doesn't throw") {
    MultiQueueThreadPool cut{1, "test_executor"};

    CATCH_CHECK_NOTHROW(cut.create_task_queue(TaskPriority::normal));
}

DEFINE_TEST("ThreadPoolQueue::push() with valid task_queue does not throw") {
    MultiQueueThreadPool cut{2, "test_executor"};
    auto& task_queue = cut.create_task_queue(TaskPriority::normal);

    CATCH_REQUIRE_NOTHROW(task_queue.push([] {}));
}

DEFINE_TEST("ThreadPoolQueue::push() with valid task_queue invokes the task") {
    MultiQueueThreadPool cut{2, "test_executor"};
    auto& task_queue = cut.create_task_queue(TaskPriority::normal);

    Flag invoked{};
    task_queue.push([&invoked] { invoked.signal(); });

    CATCH_REQUIRE(invoked.wait_for(TIMEOUT));
}

DEFINE_TEST("ThreadPoolQueue::push() invokes task on separate thread") {
    MultiQueueThreadPool cut{1, "test_executor"};
    auto& task_queue = cut.create_task_queue(TaskPriority::normal);

    Flag thread_id_assigned{};

    auto invocation_thread{std::this_thread::get_id()};

    task_queue.push([&thread_id_assigned, &invocation_thread] {
        invocation_thread = std::this_thread::get_id();
        thread_id_assigned.signal();
    });

    CATCH_CHECK(thread_id_assigned.wait_for(TIMEOUT));
    CATCH_REQUIRE(invocation_thread != std::this_thread::get_id());
}

DEFINE_TEST("MultiQueueThreadPool::join() with 2 active threads completes") {
    constexpr std::size_t num_threads{2};
    MultiQueueThreadPool cut{num_threads, "test_executor"};
    auto& task_queue = cut.create_task_queue(TaskPriority::normal);
    Flag release_busy_tasks{};
    Latch all_busy_tasks_started{num_threads};
    auto producer_threads = create_producer_threads(task_queue, num_threads,
                                                    [&release_busy_tasks, &all_busy_tasks_started] {
                                                        all_busy_tasks_started.count_down();
                                                        release_busy_tasks.wait();
                                                    });
    auto join_producer_threads = PostCondition([&producer_threads, &release_busy_tasks] {
        release_busy_tasks.signal();
        for (auto& producer_thread : producer_threads) {
            if (producer_thread.joinable()) {
                producer_thread.join();
            }
        }
    });

    // Once we know all the pool threads are busy try to join
    all_busy_tasks_started.wait();
    Flag joined_flag{};
    producer_threads.emplace_back([&cut, &joined_flag] {
        cut.join();
        joined_flag.signal();
    });

    // Check the join is blocked waiting on the busy threads
    CATCH_CHECK_FALSE(joined_flag.wait_for(FAST_TIMEOUT));

    release_busy_tasks.signal();

    CATCH_REQUIRE(joined_flag.wait_for(TIMEOUT));
}

DEFINE_TEST_FIXTURE_METHOD(
        "ThreadPoolQueue::push() high priority with pool size 2 and 2 busy normal tasks then high "
        "priority is "
        "invoked") {
    auto& normal_task_queue = cut->create_task_queue(TaskPriority::normal);
    normal_task_queue.push(create_task(0));
    normal_task_queue.push(create_task(1));

    auto& high_task_queue = cut->create_task_queue(TaskPriority::high);
    high_task_queue.push(create_task(2));

    CATCH_REQUIRE(task_started_flags[2]->wait_for(TIMEOUT));
}

}  // namespace dorado::utils::concurrency::multi_queue_thread_pool