#include "utils/AsyncQueue.h"

#include "utils/concurrency/synchronisation.h"

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <spdlog/spdlog.h>

#include <atomic>
#include <numeric>
#include <thread>

using dorado::utils::AsyncQueue;
using dorado::utils::AsyncQueueStatus;

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

// Spawned thread sits waiting for an item.
// Main thread supplies that item.
CATCH_TEST_CASE(TEST_GROUP ": PopFromOtherThread") {
    AsyncQueue<int> queue(1);
    std::atomic_bool thread_started{false};
    AsyncQueueStatus pop_status;

    auto popping_thread = std::thread([&]() {
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

    auto popping_thread = std::thread([&]() {
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
    dorado::utils::concurrency::Latch latch(num_producers + num_consumers);
    std::vector<std::size_t> processed_counts(num_consumers);

    // Start the threads.
    std::vector<std::thread> threads;
    threads.reserve(num_producers + num_consumers);
    for (int i = 0; i < num_producers; i++) {
        threads.emplace_back([&latch, &queue] {
            latch.count_down();
            latch.wait();

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
            latch.count_down();
            latch.wait();

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
    for (auto &thread : threads) {
        thread.join();
    }

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
