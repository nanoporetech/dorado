#include "utils/AsyncQueue.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#define TEST_GROUP "AsyncQueue "

#include <atomic>
#include <numeric>
#include <thread>

using dorado::utils::AsyncQueue;
using dorado::utils::AsyncQueueStatus;

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
