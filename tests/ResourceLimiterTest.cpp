#include "utils/ResourceLimiter.h"

#include "utils/concurrency/synchronisation.h"
#include "utils/jthread.h"

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <random>
#include <thread>

#define CUT_TAG "[dorado::utils::ResourceLimiter]"
#define DEFINE_TEST(name) CATCH_TEST_CASE(CUT_TAG " " name, CUT_TAG)

namespace {

using namespace dorado;
using ResourceLimiter = dorado::utils::ResourceLimiter;

DEFINE_TEST("Waiters are ordered") {
    constexpr std::size_t num_waiters = 10;
    ResourceLimiter::WaiterState waiters[num_waiters], main_waiter;
    ResourceLimiter limiter(1);

    // We use 0 as "not set", so start at 1.
    std::atomic<std::size_t> counter = 1;
    std::size_t ordering[num_waiters] = {};

    utils::concurrency::Flag ready[num_waiters];
    utils::jthread threads[num_waiters];
    for (std::size_t i = 0; i < num_waiters; i++) {
        threads[i] = utils::jthread([&, i] {
            // Wait for this thread to be signalled.
            ready[i].wait();

            // Add to the queue of allocations, which should be in thread signal order.
            {
                ResourceLimiter::ScopedReservation scope(limiter, waiters[i], 1);
                // Store what order we came in.
                ordering[i] = counter.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    // Signal the threads but take all the resources so that they're blocked on us.
    {
        ResourceLimiter::ScopedReservation scope(limiter, main_waiter, 1);
        for (auto &flag : ready) {
            // This *should* make them wait in order, but might not if the thread is woken up
            // and immediately context switches, so wait a bit too.
            flag.signal();
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        // All threads should still be blocked.
        CATCH_CHECK(limiter.sample_stats().num_waiting == num_waiters);
        for (std::size_t i = 0; i < num_waiters; i++) {
            CATCH_CAPTURE(i);
            CATCH_CHECK(ordering[i] == 0);
        }
    }

    // Wait for them to finish.
    for (auto &thread : threads) {
        thread.join();
    }

    // Check that they were in the order we expected to see.
    for (std::size_t i = 0; i < num_waiters; i++) {
        CATCH_CAPTURE(i);
        CATCH_CHECK(ordering[i] == i + 1);
    }
}

DEFINE_TEST("Multiple waiters can acquire at the same time") {
    constexpr std::size_t num_waiters = 10;
    ResourceLimiter::WaiterState waiters[num_waiters], main_waiter;
    ResourceLimiter limiter(num_waiters);

    utils::concurrency::Flag flag;
    utils::concurrency::Latch done(num_waiters);
    std::atomic_bool did_reserve = false;

    utils::jthread threads[num_waiters];
    for (std::size_t i = 0; i < num_waiters; i++) {
        threads[i] = utils::jthread([&, i] {
            // Wait for the signal.
            flag.wait();

            {
                ResourceLimiter::ScopedReservation scope(limiter, waiters[i], 1);
                did_reserve.store(true, std::memory_order_relaxed);

                // Wait for all threads to have an allocation.
                done.count_down();
                done.wait();
            }
        });
    }

    // Signal the threads but take all the resources so that they're blocked by us.
    {
        ResourceLimiter::ScopedReservation scope(limiter, main_waiter, num_waiters);
        flag.signal();

        // None of the threads should be able to make a reservation until we release ours.
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        CATCH_CHECK_FALSE(did_reserve.load(std::memory_order_relaxed));
        CATCH_CHECK(limiter.sample_stats().num_waiting == num_waiters);
    }

    // Wait for them to finish.
    for (auto &thread : threads) {
        thread.join();
    }
}

DEFINE_TEST("Limits are imposed") {
    const std::size_t max_size = GENERATE(10, 12, 19, 42);
    const std::size_t size_per_waiter = GENERATE(1, 3, 5, 10);
    CATCH_CAPTURE(max_size, size_per_waiter);

    // All waiters must reserve less than the maximum size otherwise we can
    // end up with a total that goes beyond the max size with only 1 waiter.
    CATCH_REQUIRE(size_per_waiter <= max_size);

    constexpr std::size_t num_waiters = 20;
    ResourceLimiter::WaiterState waiters[num_waiters], main_waiter;
    ResourceLimiter limiter(max_size);

    utils::concurrency::Flag flag;
    std::atomic<std::size_t> currently_reserved = 0;
    std::atomic<std::size_t> max_reserved = 0;

    utils::jthread threads[num_waiters];
    for (std::size_t i = 0; i < num_waiters; i++) {
        threads[i] = utils::jthread([&, i] {
            // Wait for the signal.
            flag.wait();

            {
                ResourceLimiter::ScopedReservation scope(limiter, waiters[i], size_per_waiter);

                // Add on our allocation.
                const auto current_total =
                        currently_reserved.fetch_add(size_per_waiter, std::memory_order_relaxed) +
                        size_per_waiter;

                // Update the max.
                auto last_max = max_reserved.load(std::memory_order_relaxed);
                while (last_max < current_total) {
                    if (max_reserved.compare_exchange_strong(last_max, current_total,
                                                             std::memory_order_relaxed)) {
                        break;
                    }
                }

                // Simulate some work.
                std::this_thread::sleep_for(std::chrono::milliseconds(10));

                // Remove our allocation.
                currently_reserved.fetch_sub(size_per_waiter, std::memory_order_relaxed);
            }
        });
    }

    // Kick off the threads and wait for them to finish.
    flag.signal();
    for (auto &thread : threads) {
        thread.join();
    }

    // Check that we only let the maximum expected through at once.
    CATCH_CHECK(max_reserved.load(std::memory_order_relaxed) <= max_size);
}

DEFINE_TEST("Not enough space is still allowed if it's empty") {
    ResourceLimiter::WaiterState waiter;
    ResourceLimiter limiter(1);
    ResourceLimiter::ScopedReservation scope(limiter, waiter, 100);
}

DEFINE_TEST("Empty allocation is safe") {
    ResourceLimiter::WaiterState waiter;
    ResourceLimiter limiter(1);
    ResourceLimiter::ScopedReservation scope(limiter, waiter, 0);
}

DEFINE_TEST("Stats report correctly") {
    ResourceLimiter::WaiterState waiters[2], main_waiter;
    ResourceLimiter limiter(10);

    auto check = [&limiter](std::size_t used, std::size_t num_waiting) {
        const auto stats = limiter.sample_stats();
        CATCH_CHECK(stats.capacity == 10);
        CATCH_CHECK(stats.used == used);
        CATCH_CHECK(stats.num_waiting == num_waiting);
    };

    CATCH_SECTION("Used count") {
        check(0, 0);
        limiter.acquire(waiters[0], 3);  // w0 +3 (3)
        check(3, 0);
        limiter.acquire(waiters[1], 5);  // w1 +5 (8)
        check(8, 0);
        limiter.release(waiters[0]);  // w0 -3 (5)
        check(5, 0);
        limiter.release(waiters[1]);  // w1 -5 (0)
        check(0, 0);
    }

    CATCH_SECTION("Blocked waiters") {
        enum class State { Waiting, Running };
        utils::concurrency::Flag ready, done;
        utils::jthread threads[2];
        for (std::size_t i = 0; i < 2; i++) {
            threads[i] = utils::jthread([&, i] {
                // Wait to be signalled.
                ready.wait();
                // Take an allocation.
                ResourceLimiter::ScopedReservation scope1(limiter, waiters[i], 8);
                // Wait to be signalled again.
                done.wait();
            });
        }

        // Nothing waiting.
        check(0, 0);

        {
            // This blocks the threads from continuing after they've woken up.
            ResourceLimiter::ScopedReservation scope1(limiter, main_waiter, 9);
            check(9, 0);
            ready.signal();

            // There's no guarantee that the threads will be blocked by our allocation
            // yet, so wait a bit before checking.
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            check(9, 2);
        }

        // After releasing our allocation, one should have made it through and
        // the other should still be waiting.
        check(8, 1);

        // Stop the threads.
        done.signal();
        for (auto &thread : threads) {
            thread.join();
        }
        check(0, 0);
    }
}

#if DORADO_ENABLE_BENCHMARK_TESTS
DEFINE_TEST("Benchmark") {
    const std::size_t num_waiters = GENERATE(1, 10);
    const std::size_t max_size = GENERATE(1, 10, 100);
    CATCH_CAPTURE(max_size, num_waiters);

    // Make some random allocation sizes to use.
    constexpr std::size_t num_allocations_per_waiter = 100;
    std::vector<std::size_t> waiter_sizes(num_waiters * num_allocations_per_waiter);
    {
        std::minstd_rand rng(42);
        std::uniform_int_distribution<std::size_t> dist(1, max_size);
        std::generate(waiter_sizes.begin(), waiter_sizes.end(), [&] { return dist(rng); });
    }

    auto waiters = std::make_unique<ResourceLimiter::WaiterState[]>(num_waiters);
    ResourceLimiter limiter(max_size);

    // Use the barriers to signal to the threads when a run starts and stops.
    std::barrier<> start(num_waiters + 1);
    std::barrier<> stop(num_waiters + 1);
    std::atomic_bool finished = false;
    std::vector<utils::jthread> threads(num_waiters);
    for (std::size_t i = 0; i < num_waiters; i++) {
        threads[i] = utils::jthread([&, i] {
            while (true) {
                // Wait for the signal.
                start.arrive_and_wait();
                if (finished.load(std::memory_order_relaxed)) {
                    break;
                }

                // Start allocating.
                for (std::size_t size : waiter_sizes) {
                    ResourceLimiter::ScopedReservation scope(limiter, waiters[i], size);
                }

                // Wait to finish so that the benchmark knows how long we took.
                stop.arrive_and_wait();
            }
        });
    }

    auto benchmark_name =
            fmt::format("ResourceLimiter: num_waiters={}, max_size={}", num_waiters, max_size);
    CATCH_BENCHMARK(std::move(benchmark_name)) {
        // Start the threads and wait for them.
        start.arrive_and_wait();
        stop.arrive_and_wait();
    };

    // Shutdown the threads.
    finished.store(true, std::memory_order_relaxed);
    start.arrive_and_wait();
}
#endif  // DORADO_ENABLE_BENCHMARK_TESTS

}  // namespace
