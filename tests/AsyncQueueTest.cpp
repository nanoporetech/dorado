#include "utils/AsyncQueue.h"

#include <catch2/catch.hpp>

#define TEST_GROUP "AsyncQueue "

#include <iostream>
#include <thread>

TEST_CASE(TEST_GROUP ": InputsMatchOutputs") {
    const int n = 10;
    AsyncQueue<int> queue(n);

    for (int i = 0; i < n; ++i) {
        const bool success = queue.try_push(std::move(i));
        REQUIRE(success);
    }
    for (int i = 0; i < n; ++i) {
        int val = -1;
        const bool success = queue.try_pop(val);
        REQUIRE(val == i);
    }
}

TEST_CASE(TEST_GROUP ": PushFailsIfTerminating") {
    AsyncQueue<int> queue(1);
    queue.terminate();
    const bool success = queue.try_push(42);
    REQUIRE(!success);
}

TEST_CASE(TEST_GROUP ": PopFailsIfTerminating") {
    AsyncQueue<int> queue(1);
    queue.terminate();
    int val;
    const bool success = queue.try_pop(val);
    REQUIRE(!success);
}

// Spawned thread sits waiting for an item.
// Main thread supplies that item.
TEST_CASE(TEST_GROUP ": PopFromOtherThread") {
    AsyncQueue<int> queue(1);
    auto popping_thread = std::thread([&queue]() {
        int val = -1;
        const bool success = queue.try_pop(val);
        REQUIRE(success);
    });

    const bool success = queue.try_push(42);
    REQUIRE(success);
    popping_thread.join();
}

// Spawned thread sits waiting for an item.
// Main thread terminates wait.
TEST_CASE(TEST_GROUP ": TerminateFromOtherThread") {
    AsyncQueue<int> queue(1);
    auto popping_thread = std::thread([&queue]() {
        int val = -1;
        const bool success = queue.try_pop(val);
        // This will fail, since the wait is terminated.
        REQUIRE(!success);
    });

    queue.terminate();
    popping_thread.join();
}