#include "utils/concurrency/no_queue_thread_pool.h"

#include "utils/PostCondition.h"
#include "utils/concurrency/synchronisation.h"

#include <catch2/catch.hpp>

#include <utility>

#define CUT_TAG "[dorado::utils::concurrency::NoQueueThreadPool]"
#define DEFINE_TEST(name) TEST_CASE(CUT_TAG " " name, CUT_TAG)
#define DEFINE_SCENARIO(name) SCENARIO(CUT_TAG " " name, CUT_TAG)
#define DEFINE_TEST_FIXTURE_METHOD(name) \
    TEST_CASE_METHOD(NoQueueThreadPoolTestFixture, CUT_TAG " " name, CUT_TAG)

using namespace std::chrono_literals;

namespace dorado::utils::concurrency::no_queue_thread_pool {

namespace {

constexpr auto TIMEOUT{10s};

std::vector<std::thread> create_producer_threads(NoQueueThreadPool& cut,
                                                 std::size_t count,
                                                 const std::function<void()>& task) {
    std::vector<std::thread> producer_threads{};
    for (std::size_t index{0}; index < count; ++index) {
        producer_threads.emplace_back([&cut, task] { cut.send(task, TaskPriority::normal); });
    }

    return producer_threads;
}

class NoQueueThreadPoolTestFixture {
protected:
    const std::size_t MAX_TASKS{4};
    const std::size_t DEFAULT_THREAD_POOL_SIZE{2};
    std::vector<std::unique_ptr<Flag>> task_release_flags{};
    std::vector<std::unique_ptr<Flag>> task_started_flags{};
    std::unique_ptr<NoQueueThreadPool> cut{};
    std::unique_ptr<std::thread> producer_thread{};

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
        cut = std::make_unique<NoQueueThreadPool>(thread_pool_size, "test_executor");
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
        if (producer_thread && producer_thread->joinable()) {
            producer_thread->join();
        }
    }
};

}  // namespace

DEFINE_TEST_FIXTURE_METHOD(
        "send() high priority with pool size 2 and 2 busy normal tasks then high priority is "
        "invoked") {
    cut->send(create_task(0), TaskPriority::normal);
    cut->send(create_task(1), TaskPriority::normal);

    producer_thread = std::make_unique<std::thread>(
            [this] { cut->send(create_task(2), TaskPriority::high); });

    REQUIRE(task_started_flags[2]->wait_for(TIMEOUT));
}

DEFINE_TEST("NoQueueThreadPool constructor with 1 thread does not throw") {
    REQUIRE_NOTHROW(NoQueueThreadPool(1));
}

DEFINE_TEST("NoQueueThreadPool::send() with task does not throw") {
    NoQueueThreadPool cut{1, "test_executor"};

    REQUIRE_NOTHROW(cut.send([] {}, TaskPriority::normal));
}

DEFINE_TEST("NoQueueThreadPool::send() with task invokes the task") {
    NoQueueThreadPool cut{1, "test_executor"};

    Flag invoked{};

    cut.send([&invoked] { invoked.signal(); }, TaskPriority::normal);

    REQUIRE(invoked.wait_for(TIMEOUT));
}

DEFINE_TEST("NoQueueThreadPool::send() invokes task on separate thread") {
    NoQueueThreadPool cut{1, "test_executor"};

    Flag thread_id_assigned{};

    auto invocation_thread{std::this_thread::get_id()};

    cut.send(
            [&thread_id_assigned, &invocation_thread] {
                invocation_thread = std::this_thread::get_id();
                thread_id_assigned.signal();
            },
            TaskPriority::normal);

    CHECK(thread_id_assigned.wait_for(TIMEOUT));
    REQUIRE(invocation_thread != std::this_thread::get_id());
}

DEFINE_TEST("NoQueueThreadPool::join() with 2 active threads completes") {
    constexpr std::size_t num_threads{2};
    NoQueueThreadPool cut{num_threads, "test_executor"};
    Flag release_busy_tasks{};
    Latch all_busy_tasks_started{num_threads};
    auto producer_threads = create_producer_threads(cut, num_threads,
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
    CHECK_FALSE(joined_flag.wait_for(200ms));

    release_busy_tasks.signal();

    REQUIRE(joined_flag.wait_for(TIMEOUT));
}

DEFINE_TEST("NoQueueThreadPool::send() when all threads busy blocks") {
    constexpr std::size_t num_threads{2};
    NoQueueThreadPool cut{num_threads, "test_executor"};
    Flag release_busy_tasks{};
    Latch all_busy_tasks_started{num_threads};
    auto producer_threads = create_producer_threads(cut, num_threads,
                                                    [&release_busy_tasks, &all_busy_tasks_started] {
                                                        all_busy_tasks_started.count_down();
                                                        release_busy_tasks.wait();
                                                    });
    auto join_producer_threads = PostCondition([&producer_threads, &release_busy_tasks] {
        release_busy_tasks.signal();
        for (auto& producer_thread : producer_threads) {
            producer_thread.join();
        }
    });

    // Once we know all the pool threads are busy enqueue another task
    all_busy_tasks_started.wait();
    Flag test_task_started{};
    producer_threads.emplace_back([&cut, &test_task_started] {
        cut.send([&test_task_started] { test_task_started.signal(); }, TaskPriority::normal);
    });

    CHECK_FALSE(test_task_started.wait_for(200ms));

    SECTION("send() is unblocked when thread can be assigned.") {
        release_busy_tasks.signal();

        CHECK(test_task_started.wait_for(TIMEOUT));
    }
}

}  // namespace dorado::utils::concurrency::no_queue_thread_pool