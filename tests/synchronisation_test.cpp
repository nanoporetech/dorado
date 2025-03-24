#include "utils/concurrency/synchronisation.h"

#include "utils/PostCondition.h"

// libtorch defines a CHECK macro, but we want catch2's version for testing
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <thread>

using namespace std::chrono_literals;
using test_clock = std::chrono::high_resolution_clock;

#define CUT_TAG "[dorado::utils::concurrency::synchronisation]"

namespace dorado::utils::concurrency::test {

namespace {
constexpr std::chrono::milliseconds TIMEOUT{2000};
}

CATCH_TEST_CASE(CUT_TAG " Latch::wait() - created with count 0 not signalled - returns", CUT_TAG) {
    Latch latch{0};
    CATCH_REQUIRE_NOTHROW(latch.wait());
    bool returned{true};
    CATCH_REQUIRE(returned);
}

CATCH_TEST_CASE(CUT_TAG " Latch::wait_for() - created with count 1 not signalled - returns false",
                CUT_TAG) {
    Latch latch{1};
    CATCH_REQUIRE_FALSE(latch.wait_for(10ms));
}

CATCH_TEST_CASE(CUT_TAG " Latch::wait_until() - created with count 1 not signalled - returns false",
                CUT_TAG) {
    Latch latch{1};
    CATCH_REQUIRE_FALSE(latch.wait_until(test_clock::now() + 10ms));
}

CATCH_TEST_CASE(CUT_TAG " Latch::count_down() - created with count 1 - no throw", CUT_TAG) {
    Latch latch{1};
    CATCH_REQUIRE_NOTHROW(latch.count_down());
}

CATCH_TEST_CASE(CUT_TAG
                " Latch::count_down() - created with count 1 already signalled once - no throw",
                CUT_TAG) {
    Latch latch{1};
    latch.count_down();
    CATCH_REQUIRE_NOTHROW(latch.count_down());
}

CATCH_TEST_CASE(CUT_TAG " Latch::wait() - created with count 1 signalled once - returns", CUT_TAG) {
    Latch latch{1};
    latch.count_down();
    CATCH_REQUIRE_NOTHROW(latch.wait());
    bool returned{true};
    CATCH_REQUIRE(returned);
}

CATCH_TEST_CASE(CUT_TAG " Latch::wait_for() - created with count 1 signalled once - returns true",
                CUT_TAG) {
    Latch latch{1};
    latch.count_down();
    CATCH_REQUIRE(latch.wait_for(10ms));
}

CATCH_TEST_CASE(CUT_TAG " Latch::wait_until() - created with count 1 signalled once - returns true",
                CUT_TAG) {
    Latch latch{1};
    latch.count_down();
    CATCH_REQUIRE(latch.wait_until(test_clock::now() + 10ms));
}

static void pause_and_signal(Latch &latch) {
    std::this_thread::sleep_for(50ms);
    latch.count_down();
}

CATCH_TEST_CASE(CUT_TAG " Latch::wait_for() - created with count 1 - block until signalled",
                CUT_TAG) {
    Latch latch{1};
    std::thread signalling_thread([&latch] { pause_and_signal(latch); });
    auto teardown = PostCondition([&signalling_thread] { signalling_thread.join(); });
    CATCH_REQUIRE(latch.wait_for(TIMEOUT));
}

CATCH_SCENARIO("Latch with count of 2", CUT_TAG) {
    Latch cut{2};
    CATCH_WHEN("signalled once") {
        std::thread first_signalling_thread([&cut] { cut.count_down(); });
        auto teardown_1 =
                PostCondition([&first_signalling_thread] { first_signalling_thread.join(); });
        CATCH_THEN("wait_for() returns false") {
            CATCH_REQUIRE_FALSE(cut.wait_for(50ms));
            CATCH_AND_WHEN("signalled a second time") {
                std::thread second_signalling_thread([&cut] { cut.count_down(); });
                auto teardown_2 = PostCondition(
                        [&second_signalling_thread] { second_signalling_thread.join(); });
                CATCH_THEN("wait_for() returns true") { CATCH_REQUIRE(cut.wait_for(TIMEOUT)); }
            }
        }
    }
}

CATCH_TEST_CASE(CUT_TAG " Flag::wait_for() - not signalled - returns false", CUT_TAG) {
    Flag flag{};
    CATCH_REQUIRE_FALSE(flag.wait_for(50ms));
}

CATCH_TEST_CASE(CUT_TAG " Flag::wait() - signalled - does not block", CUT_TAG) {
    Flag flag{};
    flag.signal();
    CATCH_REQUIRE_NOTHROW(flag.wait());
}

CATCH_TEST_CASE(CUT_TAG " Flag::wait_for() - not signalled - blocks until signalled", CUT_TAG) {
    Flag flag{};
    std::thread signalling_thread([&flag] {
        std::this_thread::sleep_for(20ms);
        flag.signal();
    });
    auto teardown = PostCondition([&signalling_thread] { signalling_thread.join(); });

    CATCH_REQUIRE_NOTHROW(flag.wait_for(TIMEOUT));
}

static void do_signal(CompositeFlag &flags, std::size_t slot) {
    std::this_thread::sleep_for(10ms);
    flags[slot].signal();
}

CATCH_TEST_CASE(CUT_TAG " CompositeFlag::wait_for() - 4 flags 3 signaled - returns false ",
                CUT_TAG) {
    CompositeFlag flags{4};
    static constexpr std::size_t NUM_THREADS{3};
    std::thread signalling_threads[NUM_THREADS];
    for (std::size_t slot{0}; slot < NUM_THREADS; ++slot) {
        signalling_threads[slot] = std::thread{[&flags, slot] { do_signal(flags, slot); }};
    }

    auto teardown = PostCondition([&signalling_threads] {
        for (std::size_t slot{0}; slot < NUM_THREADS; ++slot) {
            if (signalling_threads[slot].joinable()) {
                signalling_threads[slot].join();
            }
        }
    });

    CATCH_REQUIRE_FALSE(flags.wait_for(100ms));
}

CATCH_TEST_CASE(
        CUT_TAG
        " CompositeFlag::wait_for() - 4 flags none signalled - returns true after 4 signalled ",
        CUT_TAG) {
    CompositeFlag flags{4};
    static constexpr std::size_t NUM_THREADS{4};
    std::thread signalling_threads[NUM_THREADS];
    for (std::size_t slot{0}; slot < NUM_THREADS; ++slot) {
        signalling_threads[slot] = std::thread{[&flags, slot] { do_signal(flags, slot); }};
    }

    auto teardown = PostCondition([&signalling_threads] {
        for (std::size_t slot{0}; slot < NUM_THREADS; ++slot) {
            if (signalling_threads[slot].joinable()) {
                signalling_threads[slot].join();
            }
        }
    });

    CATCH_REQUIRE(flags.wait_for(TIMEOUT));
}

CATCH_TEST_CASE(
        CUT_TAG
        " Flag::wait() - single producer multiple consumers - all consumers block until producer "
        "signals",
        CUT_TAG) {
    Flag producer_flag{};
    static constexpr std::size_t NUM_CONSUMERS{4};
    Latch consumers_started{NUM_CONSUMERS};
    Latch consumers_signalled{NUM_CONSUMERS};

    std::thread consumer_threads[NUM_CONSUMERS];

    auto teardown = PostCondition([&consumer_threads] {
        for (std::size_t slot{0}; slot < NUM_CONSUMERS; ++slot) {
            if (consumer_threads[slot].joinable()) {
                consumer_threads[slot].join();
            }
        }
    });

    for (std::size_t thread_index{0}; thread_index < NUM_CONSUMERS; ++thread_index) {
        consumer_threads[thread_index] =
                std::thread{[&producer_flag, &consumers_started, &consumers_signalled] {
                    consumers_started.count_down();
                    if (producer_flag.wait_for(TIMEOUT)) {
                        consumers_signalled.count_down();
                    }
                }};
    }

    CATCH_REQUIRE(consumers_started.wait_for(TIMEOUT));
    std::this_thread::sleep_for(10ms);  // give consumers a chance to become blocked in wait()
    producer_flag.signal();
    CATCH_REQUIRE(consumers_signalled.wait_for(TIMEOUT));
}

CATCH_TEST_CASE(CUT_TAG " TSan data race in Latch", CUT_TAG) {
    // Spin up a runner that will wait for a Flag it should signal
    std::atomic<Flag *> flag_ptr{nullptr};
    std::thread runner([&flag_ptr] {
        while (true) {
            auto *flag = flag_ptr.load(std::memory_order_acquire);
            if (flag != nullptr) {
                flag->signal();
                break;
            }
            std::this_thread::yield();
        }
    });

    {
        // Construct a new Flag and hand it to the runner to signal.
        auto flushed_flag = std::make_unique<Flag>();
        flag_ptr.store(flushed_flag.get(), std::memory_order_release);

        // Wait long enough that the runner should have signalled it.
        // This way the call to wait() should return immediately without
        // locking the Latch's condition_variable's internal mutex.
        std::this_thread::sleep_for(TIMEOUT);
        flushed_flag->wait();

        // The condition_variable's destructor touches its internal
        // mutex in order to destroy it, which will trip TSan's data
        // race detection since it was last written to by the runner
        // and hasn't been synchronised-with on this thread.
        flushed_flag.reset();
    }

    runner.join();
}

}  // namespace dorado::utils::concurrency::test
