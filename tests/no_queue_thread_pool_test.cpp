#include "utils/concurrency/no_queue_thread_pool.h"

#include "utils/PostCondition.h"
#include "utils/concurrency/synchronisation.h"

#include <catch2/catch.hpp>

#include <utility>

#define CUT_TAG "[dorado::utils::concurrency::NoQueueThreadPool]"
#define DEFINE_TEST(name) TEST_CASE(CUT_TAG " " name, CUT_TAG)

using namespace std::chrono_literals;

namespace dorado::utils::concurrency::no_queue_thread_pool {

namespace {
constexpr auto TIMEOUT{10s};

std::vector<std::thread> create_producer_threads(NoQueueThreadPool& cut,
                                                 std::size_t count,
                                                 const std::function<void()>& task) {
    std::vector<std::thread> producer_threads{};
    for (std::size_t index{0}; index < count; ++index) {
        producer_threads.emplace_back([&cut, task] { cut.send(task); });
    }

    return producer_threads;
}

}  // namespace

DEFINE_TEST("NoQueueThreadPool constructor with 1 thread does not throw") {
    REQUIRE_NOTHROW(NoQueueThreadPool(1));
}

DEFINE_TEST("NoQueueThreadPool::send() with task does not throw") {
    NoQueueThreadPool cut{1, "test_executor"};

    REQUIRE_NOTHROW(cut.send([] {}));
}

DEFINE_TEST("NoQueueThreadPool::send() with task invokes the task") {
    NoQueueThreadPool cut{1, "test_executor"};

    Flag invoked{};

    cut.send([&invoked] { invoked.signal(); });

    REQUIRE(invoked.wait_for(TIMEOUT));
}

DEFINE_TEST("NoQueueThreadPool::send() invokes task on separate thread") {
    NoQueueThreadPool cut{1, "test_executor"};

    Flag thread_id_assigned{};

    auto invocation_thread{std::this_thread::get_id()};

    cut.send([&thread_id_assigned, &invocation_thread] {
        invocation_thread = std::this_thread::get_id();
        thread_id_assigned.signal();
    });

    CHECK(thread_id_assigned.wait_for(TIMEOUT));
    REQUIRE(invocation_thread != std::this_thread::get_id());
}

DEFINE_TEST("NoQueueThreadPool::join() with 2 active threads completes") {
    constexpr std::size_t num_threads{2};
    NoQueueThreadPool cut{num_threads, "test_executor"};
    Flag release_busy_tasks{};
    auto producer_threads = create_producer_threads(
            cut, num_threads, [&release_busy_tasks] { release_busy_tasks.wait(); });
    auto join_producer_threads = PostCondition([&producer_threads, &release_busy_tasks] {
        release_busy_tasks.signal();
        for (auto& producer_thread : producer_threads) {
            if (producer_thread.joinable()) {
                producer_thread.join();
            }
        }
    });

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
        cut.send([&test_task_started] { test_task_started.signal(); });
    });

    CHECK_FALSE(test_task_started.wait_for(200ms));

    SECTION("send() is unblocked when thread can be assigned.") {
        release_busy_tasks.signal();

        CHECK(test_task_started.wait_for(TIMEOUT));
    }
}

}  // namespace dorado::utils::concurrency::no_queue_thread_pool