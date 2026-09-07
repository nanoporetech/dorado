#include "utils/AsyncQueue.h"

#include "utils/jthread.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <atomic>
#include <chrono>
#include <numeric>
#include <thread>

#if DORADO_ENABLE_BENCHMARK_TESTS
#include <spdlog/spdlog.h>

#include <latch>
#endif

using dorado::utils::AsyncQueue;
using dorado::utils::AsyncQueueStatus;
using dorado::utils::AsyncQueueNonBlockingMode::FullLock;
using dorado::utils::AsyncQueueNonBlockingMode::TryLock;

#define TEST_GROUP "AsyncQueue "

CATCH_TEST_CASE(TEST_GROUP ": InputsMatchOutputs") {
    const int n = 10;
    AsyncQueue<int> queue(n);

    for (int i = 0; i < n; ++i) {
        // clang-tidy don't like us reusing a moved-from variable even if it's trivial,
        // so store to a temporary that's not used again after it's moved.
        int ii = i;
        const auto status = queue.try_push(std::move(ii));
        CATCH_REQUIRE(status == AsyncQueueStatus::Success);
    }
    for (int i = 0; i < n; ++i) {
        int val = -1;
        const auto status = queue.try_pop(val);
        CATCH_REQUIRE(status == AsyncQueueStatus::Success);
        CATCH_CHECK(val == i);
    }
}

CATCH_TEST_CASE(TEST_GROUP ": PushFailsIfTerminating") {
    const auto terminate_mode = GENERATE(dorado::utils::AsyncQueueTerminateFast::No,
                                         dorado::utils::AsyncQueueTerminateFast::Yes);

    AsyncQueue<int> queue(1);
    queue.terminate(terminate_mode);
    const auto status = queue.try_push(42);
    CATCH_CHECK(status == AsyncQueueStatus::Terminate);
}

CATCH_TEST_CASE(TEST_GROUP ": PopFailsIfTerminatingFast") {
    AsyncQueue<int> queue(1);
    auto status = queue.try_push(42);
    CATCH_CHECK(status == AsyncQueueStatus::Success);
    queue.terminate(dorado::utils::AsyncQueueTerminateFast::Yes);
    int val = 0;
    status = queue.try_pop(val);
    CATCH_CHECK(status == AsyncQueueStatus::Terminate);
}

CATCH_TEST_CASE(TEST_GROUP ": PopSucceedsIfTerminatingSlow") {
    AsyncQueue<int> queue(1);
    auto status = queue.try_push(42);
    CATCH_CHECK(status == AsyncQueueStatus::Success);
    queue.terminate(dorado::utils::AsyncQueueTerminateFast::No);
    int val = 0;
    status = queue.try_pop(val);
    CATCH_CHECK(status == AsyncQueueStatus::Success);
    CATCH_CHECK(val == 42);
    status = queue.try_pop(val);
    CATCH_CHECK(status == AsyncQueueStatus::Terminate);
}

CATCH_TEST_CASE(TEST_GROUP ": PushPopSucceedAfterRestarting") {
    const auto terminate_mode = GENERATE(dorado::utils::AsyncQueueTerminateFast::No,
                                         dorado::utils::AsyncQueueTerminateFast::Yes);

    AsyncQueue<int> queue(1);
    queue.terminate(terminate_mode);
    queue.restart();
    const auto push_status = queue.try_push(42);
    CATCH_CHECK(push_status == AsyncQueueStatus::Success);
    int val = 0;
    const auto pop_status = queue.try_pop(val);
    CATCH_CHECK(pop_status == AsyncQueueStatus::Success);
    CATCH_CHECK(val == 42);
}

CATCH_TEST_CASE(TEST_GROUP ": QueueEmptyAfterRestarting") {
    const auto terminate_mode = GENERATE(dorado::utils::AsyncQueueTerminateFast::No,
                                         dorado::utils::AsyncQueueTerminateFast::Yes);

    AsyncQueue<int> queue(5);
    CATCH_CHECK(queue.try_push(1) == AsyncQueueStatus::Success);
    CATCH_CHECK(queue.try_push(2) == AsyncQueueStatus::Success);
    CATCH_CHECK(queue.try_push(3) == AsyncQueueStatus::Success);
    queue.terminate(terminate_mode);
    queue.restart();
    CATCH_CHECK(queue.size() == 0);
}

CATCH_TEST_CASE(TEST_GROUP ": NonBlocking") {
    AsyncQueue<int> queue(5);
    CATCH_CHECK(queue.try_push(1) == AsyncQueueStatus::Success);
    CATCH_CHECK(queue.try_push(2) == AsyncQueueStatus::Success);

    // Pop the items.
    int val = 0;
    CATCH_CHECK(queue.try_pop_nonblocking(val, FullLock) == AsyncQueueStatus::Success);
    CATCH_CHECK(val == 1);
    {
        // Blocking the queue should cause a timeout rather than blocking the caller.
        auto blocker = queue.block_for_testing();
        CATCH_CHECK(queue.try_pop_nonblocking(val, TryLock) == AsyncQueueStatus::Timeout);
    }
    CATCH_CHECK(queue.try_pop_nonblocking(val, FullLock) == AsyncQueueStatus::Success);
    CATCH_CHECK(val == 2);

    // Popping from an empty queue should timeout too.
    CATCH_CHECK(queue.try_pop_nonblocking(val, FullLock) == AsyncQueueStatus::Timeout);
}

CATCH_TEST_CASE(TEST_GROUP ": try_pop_until") {
    using Clock = AsyncQueue<int>::Clock;

    AsyncQueue<int> queue(5);
    CATCH_CHECK(queue.try_push(1) == AsyncQueueStatus::Success);
    CATCH_CHECK(queue.try_push(2) == AsyncQueueStatus::Success);

    const auto small_timeout = std::chrono::milliseconds(200);
    const auto distant_future = std::chrono::seconds(100);

    // Pop the items.
    int val = 0;
    CATCH_CHECK(queue.try_pop_until(val, Clock::now() + distant_future) ==
                AsyncQueueStatus::Success);
    CATCH_CHECK(val == 1);
    CATCH_CHECK(queue.try_pop_until(val, Clock::now() + small_timeout) ==
                AsyncQueueStatus::Success);
    CATCH_CHECK(val == 2);

    // Popping from an empty queue should timeout.
    CATCH_CHECK(queue.try_pop_until(val, Clock::now() + small_timeout) ==
                AsyncQueueStatus::Timeout);
}

// Spawned thread sits waiting for an item.
// Main thread supplies that item.
CATCH_TEST_CASE(TEST_GROUP ": PopFromOtherThread") {
    AsyncQueue<int> queue(1);
    std::atomic_bool thread_started{false};
    AsyncQueueStatus pop_status;

    auto popping_thread = dorado::utils::jthread([&]() {
        thread_started.store(true);
        int val = -1;
        // catch2 isn't thread safe so we have to check this on the main thread
        pop_status = queue.try_pop(val);
    });

    // Wait for thread to start
    while (!thread_started.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Feed data to the thread
    const auto push_status = queue.try_push(42);
    CATCH_CHECK(push_status == AsyncQueueStatus::Success);

    popping_thread.join();
    CATCH_CHECK(pop_status == AsyncQueueStatus::Success);
}

// Spawned thread sits waiting for an item.
// Main thread terminates wait.
CATCH_TEST_CASE(TEST_GROUP ": TerminateFromOtherThread") {
    AsyncQueue<int> queue(1);
    std::atomic_bool thread_started{false};
    AsyncQueueStatus pop_status;

    auto popping_thread = dorado::utils::jthread([&]() {
        thread_started.store(true);
        int val = -1;
        // catch2 isn't thread safe so we have to check this on the main thread
        pop_status = queue.try_pop(val);
    });

    // Wait for thread to start
    while (!thread_started.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Stop it
    queue.terminate(dorado::utils::AsyncQueueTerminateFast::No);
    popping_thread.join();

    // This will fail, since the wait is terminated.
    CATCH_CHECK(pop_status == AsyncQueueStatus::Terminate);
}

CATCH_TEST_CASE(TEST_GROUP ": process_and_pop_n") {
    const int n = 10;
    AsyncQueue<int> queue(n);
    for (int i = 0; i < n; ++i) {
        // clang-tidy don't like us reusing a moved-from variable even if it's trivial,
        // so store to a temporary that's not used again after it's moved.
        int ii = i;
        const auto status = queue.try_push(std::move(ii));
        CATCH_REQUIRE(status == AsyncQueueStatus::Success);
    }

    std::vector<int> popped_items;
    auto pop_item = [&popped_items](int popped) { popped_items.push_back(popped); };

    // Pop 5 of the items.
    auto status = queue.process_and_pop_n(pop_item, 5);
    CATCH_REQUIRE(status == AsyncQueueStatus::Success);
    CATCH_CHECK(popped_items.size() == 5);
    CATCH_CHECK(queue.size() == 5);

    // Pop the other 5 items.
    status = queue.process_and_pop_n(pop_item, 5);
    CATCH_REQUIRE(status == AsyncQueueStatus::Success);

    std::vector<int> expected(n);
    std::iota(expected.begin(), expected.end(), 0);
    CATCH_CHECK(popped_items == expected);
    CATCH_CHECK(queue.size() == 0);
}

CATCH_TEST_CASE(TEST_GROUP ": process_and_pop_n_with_timeout") {
    using Clock = AsyncQueue<int>::Clock;

    AsyncQueue<int> queue(5);
    CATCH_CHECK(queue.try_push(1) == AsyncQueueStatus::Success);
    CATCH_CHECK(queue.try_push(2) == AsyncQueueStatus::Success);
    CATCH_CHECK(queue.try_push(3) == AsyncQueueStatus::Success);
    CATCH_CHECK(queue.try_push(4) == AsyncQueueStatus::Success);

    const auto small_timeout = std::chrono::milliseconds(200);
    const auto distant_future = std::chrono::seconds(100);

    std::vector<int> popped_items;
    auto pop_item = [&popped_items](int popped) { popped_items.push_back(popped); };

    // Take 2 of the 4 items.
    CATCH_CHECK(queue.process_and_pop_n_with_timeout(pop_item, 2, Clock::now() + distant_future) ==
                AsyncQueueStatus::Success);
    CATCH_REQUIRE(popped_items.size() == 2);
    CATCH_CHECK(popped_items.at(0) == 1);
    CATCH_CHECK(popped_items.at(1) == 2);

    // Try and take 3 of the remaining 2 items.
    // The timeout is on waiting for any elements, so we should pop 2 successfully.
    popped_items.clear();
    CATCH_CHECK(queue.process_and_pop_n_with_timeout(pop_item, 3, Clock::now() + small_timeout) ==
                AsyncQueueStatus::Success);
    CATCH_REQUIRE(popped_items.size() == 2);
    CATCH_CHECK(popped_items.at(0) == 3);
    CATCH_CHECK(popped_items.at(1) == 4);

    // Popping from an empty queue should timeout.
    popped_items.clear();
    CATCH_CHECK(queue.process_and_pop_n_with_timeout(pop_item, 1, Clock::now() + small_timeout) ==
                AsyncQueueStatus::Timeout);
    CATCH_CHECK(popped_items.empty());
}

CATCH_TEST_CASE(TEST_GROUP ": name") {
    AsyncQueue<int> queue(1);
    CATCH_CHECK(queue.get_name() == "queue");
    queue.set_name("test");
    CATCH_CHECK(queue.get_name() == "test");
}

#if DORADO_ENABLE_BENCHMARK_TESTS
CATCH_TEST_CASE(TEST_GROUP ": benchmarks") {
    const int run_for_ms = 2'500;

    const bool unbounded = GENERATE(true, false);
    const int num_producers = GENERATE(1, 2, 4);
    const int num_consumers = GENERATE(1, 2, 4);

    using Item = std::unique_ptr<int>;
    AsyncQueue<Item> queue(unbounded ? 1'000'000 : 10);
    std::latch latch(num_producers + num_consumers);
    std::vector<std::size_t> processed_counts(num_consumers);

    // Start the threads.
    std::vector<dorado::utils::jthread> threads;
    threads.reserve(num_producers + num_consumers);
    for (int i = 0; i < num_producers; i++) {
        threads.emplace_back([&latch, &queue] {
            latch.arrive_and_wait();

            while (true) {
                auto res = queue.try_push(Item{});
                if (res == AsyncQueueStatus::Terminate) {
                    break;
                }
            }
        });
    }
    for (int i = 0; i < num_consumers; i++) {
        auto &counter = processed_counts.at(i);
        threads.emplace_back([&latch, &queue, &counter] {
            latch.arrive_and_wait();

            std::size_t processed = 0;
            while (true) {
                Item item;
                auto res = queue.try_pop(item);
                if (res == AsyncQueueStatus::Terminate) {
                    break;
                }
                processed++;
            }

            counter = processed;
        });
    }

    // Wait for the threads to start, then let them run for a bit.
    latch.wait();
    std::this_thread::sleep_for(std::chrono::milliseconds(run_for_ms));
    queue.terminate(dorado::utils::AsyncQueueTerminateFast::Yes);
    threads.clear();

    // Collect timings.
    const double total_processed =
            std::accumulate(processed_counts.begin(), processed_counts.end(), std::size_t{0});
    const double speed = total_processed * 1000.0 / run_for_ms;
    spdlog::info(TEST_GROUP
                 ": Speed for unbounded={}, producers={}, consumers={}: {:.2e} items in {}ms "
                 "({:.2e}items/s)",
                 unbounded, num_producers, num_consumers, total_processed, run_for_ms, speed);
}
#endif
