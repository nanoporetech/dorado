#include "utils/concurrency/async_task_executor.h"

#include "utils/PostCondition.h"
#include "utils/concurrency/synchronisation.h"

#include <catch2/catch.hpp>

#include <utility>

#define CUT_TAG "[dorado::utils::concurrency::AsyncTaskExecutor]"
#define DEFINE_TEST(name) TEST_CASE(CUT_TAG " " name, CUT_TAG)

using namespace std::chrono_literals;

namespace dorado::utils::concurrency::async_task_executor {

namespace {
constexpr auto TIMEOUT{5s};

std::vector<std::thread> create_producer_threads(AsyncTaskExecutor& cut,
                                                 std::size_t count,
                                                 std::function<void()> task) {
    std::vector<std::thread> producer_threads{};
    for (std::size_t index{0}; index < count; ++index) {
        producer_threads.emplace_back([&cut, task] { cut.send(task); });
    }

    return producer_threads;
}

}  // namespace

DEFINE_TEST("Constructor with 1 thread does not throw") { REQUIRE_NOTHROW(AsyncTaskExecutor(1)); }

DEFINE_TEST("send() with task does not thread") {
    AsyncTaskExecutor cut{1, "test_executor"};

    REQUIRE_NOTHROW(cut.send([] {}));
}

DEFINE_TEST("send() with task invokes the task") {
    AsyncTaskExecutor cut{1, "test_executor"};

    Flag invoked{};

    cut.send([&invoked] { invoked.signal(); });

    REQUIRE(invoked.wait_for(TIMEOUT));
}

DEFINE_TEST("send() invokes task on separate thread") {
    AsyncTaskExecutor cut{1, "test_executor"};

    Flag thread_id_assigned{};

    auto invocation_thread{std::this_thread::get_id()};

    cut.send([&thread_id_assigned, &invocation_thread] {
        invocation_thread = std::this_thread::get_id();
        thread_id_assigned.signal();
    });

    CHECK(thread_id_assigned.wait_for(TIMEOUT));
    REQUIRE(invocation_thread != std::this_thread::get_id());
}

DEFINE_TEST("join() with 2 active threads completes") {
    constexpr std::size_t num_threads{2};
    AsyncTaskExecutor cut{num_threads, "test_executor"};
    Flag release_busy_tasks{};
    auto producer_threads = create_producer_threads(
            cut, num_threads, [&release_busy_tasks] { release_busy_tasks.wait(); });
    auto join_producer_threads = PostCondition([&producer_threads, &release_busy_tasks] {
        release_busy_tasks.signal();
        for (auto& producer_thread : producer_threads) {
            producer_thread.join();
        }
    });

    Flag joined_flag{};
    producer_threads.emplace_back([&cut, &joined_flag] {
        cut.join();
        joined_flag.signal();
    });

    // Check the join is blocked waiting on the busy threads
    CHECK_FALSE(joined_flag.wait_for(1s));

    release_busy_tasks.signal();

    REQUIRE(joined_flag.wait_for(TIMEOUT));
}

DEFINE_TEST("send() when all threads busy blocks") {
    constexpr std::size_t num_threads{2};
    AsyncTaskExecutor cut{num_threads, "test_executor"};
    Flag release_busy_tasks{};
    auto producer_threads = create_producer_threads(
            cut, num_threads, [&release_busy_tasks] { release_busy_tasks.wait(); });
    auto join_producer_threads = PostCondition([&producer_threads, &release_busy_tasks] {
        release_busy_tasks.signal();
        for (auto& producer_thread : producer_threads) {
            producer_thread.join();
        }
    });

    Flag test_task_started{};
    producer_threads.emplace_back([&cut, &test_task_started] {
        cut.send([&test_task_started] { test_task_started.signal(); });
    });

    CHECK_FALSE(test_task_started.wait_for(1s));

    SECTION("send() is unblocked when thread can be assigned.") {
        release_busy_tasks.signal();

        CHECK(test_task_started.wait_for(TIMEOUT));
    }
}

}  // namespace dorado::utils::concurrency::async_task_executor