#include "utils/AsyncQueue.h"

#include <catch2/catch_test_macros.hpp>

#define TEST_GROUP "AsyncQueue "

#include <algorithm>
#include <atomic>
#include <iostream>
#include <numeric>
#include <thread>

using dorado::utils::AsyncQueue;
using dorado::utils::AsyncQueueStatus;

TEST_CASE(TEST_GROUP ": InputsMatchOutputs") {
    const int n = 10;
    AsyncQueue<int> queue(n);

    for (int i = 0; i < n; ++i) {
        // clang-tidy don't like us reusing a moved-from variable even if it's trivial,
        // so store to a temporary that's not used again after it's moved.
        int ii = i;
        const auto status = queue.try_push(std::move(ii));
        REQUIRE(status == AsyncQueueStatus::Success);
    }
    for (int i = 0; i < n; ++i) {
        int val = -1;
        const auto status = queue.try_pop(val);
        REQUIRE(status == AsyncQueueStatus::Success);
        CHECK(val == i);
    }
}

TEST_CASE(TEST_GROUP ": PushFailsIfTerminating") {
    AsyncQueue<int> queue(1);
    queue.terminate();
    const auto status = queue.try_push(42);
    CHECK(status == AsyncQueueStatus::Terminate);
}

TEST_CASE(TEST_GROUP ": PopFailsIfTerminating") {
    AsyncQueue<int> queue(1);
    queue.terminate();
    int val;
    const auto status = queue.try_pop(val);
    CHECK(status == AsyncQueueStatus::Terminate);
}

TEST_CASE(TEST_GROUP ": PushPopSucceedAfterRestarting") {
    AsyncQueue<int> queue(1);
    queue.terminate();
    queue.restart();
    const auto push_status = queue.try_push(42);
    CHECK(push_status == AsyncQueueStatus::Success);
    int val;
    const auto pop_status = queue.try_pop(val);
    CHECK(pop_status == AsyncQueueStatus::Success);
}

// Spawned thread sits waiting for an item.
// Main thread supplies that item.
TEST_CASE(TEST_GROUP ": PopFromOtherThread") {
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
    CHECK(push_status == AsyncQueueStatus::Success);

    popping_thread.join();
    CHECK(pop_status == AsyncQueueStatus::Success);
}

// Spawned thread sits waiting for an item.
// Main thread terminates wait.
TEST_CASE(TEST_GROUP ": TerminateFromOtherThread") {
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
    queue.terminate();
    popping_thread.join();

    // This will fail, since the wait is terminated.
    CHECK(pop_status == AsyncQueueStatus::Terminate);
}

TEST_CASE(TEST_GROUP ": process_and_pop_n") {
    const int n = 10;
    AsyncQueue<int> queue(n);
    for (int i = 0; i < n; ++i) {
        // clang-tidy don't like us reusing a moved-from variable even if it's trivial,
        // so store to a temporary that's not used again after it's moved.
        int ii = i;
        const auto status = queue.try_push(std::move(ii));
        REQUIRE(status == AsyncQueueStatus::Success);
    }

    std::vector<int> popped_items;
    auto pop_item = [&popped_items](int popped) { popped_items.push_back(popped); };

    // Pop 5 of the items.
    auto status = queue.process_and_pop_n(pop_item, 5);
    REQUIRE(status == AsyncQueueStatus::Success);
    CHECK(popped_items.size() == 5);
    CHECK(queue.size() == 5);

    // Pop the other 5 items.
    status = queue.process_and_pop_n(pop_item, 5);
    REQUIRE(status == AsyncQueueStatus::Success);

    std::vector<int> expected(n);
    std::iota(expected.begin(), expected.end(), 0);
    CHECK(popped_items == expected);
    CHECK(queue.size() == 0);
}