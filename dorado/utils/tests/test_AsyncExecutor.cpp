#include "utils/concurrency/AsyncExecutor.h"
#include "utils/concurrency/TaskPool.h"
#include "utils/concurrency/WorkerPool.h"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <latch>
#include <optional>
#include <thread>

#if DORADO_ENABLE_BENCHMARK_TESTS
#include "utils/jthread.h"

#include <catch2/benchmark/catch_benchmark.hpp>
#include <spdlog/fmt/bundled/format.h>
#include <spdlog/spdlog.h>

#include <random>
#endif

#define CUT_TAG "[AsyncExecutor]"
#define DEFINE_TEST(name) CATCH_TEST_CASE(CUT_TAG " " name, CUT_TAG)
#define DEFINE_TEMPLATE_TEST(name, ...) \
    CATCH_TEMPLATE_TEST_CASE(CUT_TAG " " name, CUT_TAG, __VA_ARGS__)

using namespace dorado::utils::concurrency;

namespace {

void busy_task(std::chrono::microseconds run_for) {
    using Clock = std::chrono::steady_clock;
    const auto end_time = Clock::now() + run_for;

    // Don't just sleep since it doesn't put any load on the system, which isn't useful for benchmarks.
    volatile float f = 123.f;
    while (Clock::now() < end_time) {
        for (std::size_t i = 0; i < 1'000; i++) {
            f = std::sqrt(f) + 1.f;
        }
    }
}

std::vector<AsyncExecutor> make_simple_producers(TaskPool &tasks, std::size_t num_producers) {
    std::vector<AsyncExecutor> producers;
    producers.reserve(num_producers);
    for (std::size_t producer_idx = 0; producer_idx < num_producers; producer_idx++) {
        producers.emplace_back(tasks, producer_idx);
    }
    return producers;
}

DEFINE_TEST("Rebind task pools") {
    // Also functions as a smoke test.
    const std::size_t num_workers = GENERATE(1, 2, 4);
    const std::size_t num_producers = GENERATE(1, 2, 4);
    const std::size_t queue_capacity = GENERATE(1, 10);
    const std::size_t num_tasks_per_producer = 10;
    const std::size_t rebind_count = 2;
    CATCH_CAPTURE(num_workers, num_producers, queue_capacity);

    // Create the workers.
    WorkerPool workers(num_workers, "test");

    std::atomic_size_t jobs_run = 0;
    for (std::size_t repeat = 0; repeat < rebind_count; repeat++) {
        // Create the task pool and bind it.
        TaskPool tasks(num_producers, queue_capacity);
        {
            WorkerPool::BindTasks binder(workers, tasks);

            // Setup producers.
            std::vector producers = make_simple_producers(tasks, num_producers);

            // Push some tasks.
            for (std::size_t task_idx = 0; task_idx < num_tasks_per_producer; task_idx++) {
                for (auto &producer : producers) {
                    producer.send([&jobs_run] {
                        busy_task(std::chrono::microseconds(100));
                        jobs_run.fetch_add(1, std::memory_order_relaxed);
                    });
                }
            }

            // Unbinding the task pool should force a flush, so no need for any explicit synchronisation.
        }

        // Check that all the jobs did run.
        CATCH_CHECK(jobs_run.load(std::memory_order_relaxed) ==
                    num_tasks_per_producer * num_producers);
        jobs_run.store(0, std::memory_order_relaxed);
    }
}

DEFINE_TEST("Limits are followed") {
    const std::size_t num_workers = GENERATE(1, 2);
    const std::size_t num_producers = 1;

    // Create the workers.
    WorkerPool workers(num_workers, "test");

    for (std::size_t queue_capacity : {1, 2, 5, 10}) {
        // The main thread will block the tasks.
        std::latch latch(1);

        // Create the task pool and bind it.
        TaskPool tasks(num_producers, queue_capacity);
        WorkerPool::BindTasks binder(workers, tasks);

        // Setup producer.
        AsyncExecutor producer(tasks, 0);

        // Push as many tasks as the pool should be able to hold.
        // The workers will each pop one of the tasks, so we push that many more.
        for (std::size_t task_id = 0; task_id < queue_capacity + num_workers; task_id++) {
            producer.send([&latch] { latch.wait(); });
        }

        // We can't push another task to the queue without it blocking, but we can check that it's full.
        CATCH_CHECK(tasks.queue_size(0) == queue_capacity);
        CATCH_CHECK(tasks.tasks_in_flight(0) == queue_capacity + num_workers);

        // Unpause the workers.
        latch.count_down();
    }
}

DEFINE_TEST("All workers take from all producers") {
    const std::size_t num_workers = 2;
    const std::size_t num_producers = 2;
    const std::size_t queue_capacity = 2;

    // Create the workers.
    WorkerPool workers(num_workers, "test");

    for (std::size_t mask = 0; mask < 4; mask++) {
        const std::size_t idx_a = (mask & 1) ? 1 : 0;
        const std::size_t idx_b = (mask & 2) ? 1 : 0;

        std::latch latch(2);

        // Create the task pool and bind it.
        TaskPool tasks(num_producers, queue_capacity);
        WorkerPool::BindTasks binder(workers, tasks);

        // Setup the producers.
        std::vector producers = make_simple_producers(tasks, num_producers);

        // Pick 2 producers to push to since the workers should try and pop from both.
        auto &producer_a = producers.at(idx_a);
        auto &producer_b = producers.at(idx_b);

        // Push the tasks, both waiting on the latch.
        producer_a.send([&latch] { latch.arrive_and_wait(); });
        producer_b.send([&latch] { latch.arrive_and_wait(); });

        // Both tasks should be able to run at the same time.
        latch.wait();
    }
}

DEFINE_TEST("Per-producer flushing works") {
    const std::size_t num_workers = 2;
    const std::size_t queue_capacity = 100;
    const std::chrono::microseconds task_time(100);

    // Create the workers.
    WorkerPool workers(num_workers, "test");

    for (std::size_t num_producers : {1, 2}) {
        // Per-producer counters.
        std::vector<std::atomic_size_t> counters(num_producers);

        // Create the task pool and bind it.
        TaskPool tasks(num_producers, queue_capacity);
        WorkerPool::BindTasks binder(workers, tasks);

        // Setup producer.
        std::vector producers = make_simple_producers(tasks, num_producers);

        auto push_reads = [&](std::size_t producer_idx) {
            auto &producer = producers[producer_idx];
            auto &counter = counters[producer_idx];
            for (std::size_t task_id = 0; task_id < queue_capacity; task_id++) {
                producer.send([&counter, task_time] {
                    busy_task(task_time);
                    counter.fetch_add(1, std::memory_order_relaxed);
                });
            }
        };

        auto flush_and_reset = [&](std::size_t producer_idx) {
            auto &producer = producers[producer_idx];
            auto &counter = counters[producer_idx];
            producer.flush();
            CATCH_REQUIRE(counter.load(std::memory_order_relaxed) == queue_capacity);
            counter.store(0, std::memory_order_relaxed);
        };

        // Check each individually.
        for (std::size_t producer_idx = 0; producer_idx < num_producers; producer_idx++) {
            push_reads(producer_idx);
            flush_and_reset(producer_idx);
        }

        // Interleave them.
        if (num_producers > 1) {
            for (std::size_t mask = 0; mask < 4; mask++) {
                const std::size_t first_push = (mask & 1) ? 1 : 0;
                const std::size_t first_flush = (mask & 2) ? 1 : 0;
                push_reads(first_push);
                push_reads(1 - first_push);
                flush_and_reset(first_flush);
                flush_and_reset(1 - first_flush);
            }
        }
    }
}

DEFINE_TEST("Bad queue index") {
    const std::size_t num_producers = 2;
    const std::size_t queue_capacity = 10;
    TaskPool tasks(num_producers, queue_capacity);

    auto make_executor = [&tasks](std::size_t q_idx) { return AsyncExecutor(tasks, q_idx); };

    CATCH_CHECK_NOTHROW(make_executor(0));
    CATCH_CHECK_NOTHROW(make_executor(1));
    CATCH_CHECK_THROWS_AS(make_executor(2), std::logic_error);
}

DEFINE_TEST("Empty pools don't crash") {
    const std::size_t num_workers = GENERATE(0, 1);
    const std::size_t num_queues = GENERATE(0, 1);
    const std::size_t queue_capacity = GENERATE(0, 1);
    const bool with_bind = GENERATE(false, true);
    const bool with_wait = GENERATE(false, true);
    CATCH_CAPTURE(num_workers, num_queues, queue_capacity, with_bind, with_wait);

    WorkerPool workers(num_workers, "test");
    TaskPool tasks(num_queues, queue_capacity);

    std::optional<WorkerPool::BindTasks> binder;
    if (with_bind) {
        binder.emplace(workers, tasks);
    }

    if (with_wait) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

#if DORADO_ENABLE_BENCHMARK_TESTS

}  // namespace

#ifndef _WIN32
#include <sys/resource.h>
#include <sys/time.h>
#endif

namespace {

// Helper to determine how much CPU has been used.
double cpu_rusage() {
#ifdef _WIN32
    return 0;
#else
    rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0;
    }
    auto to_seconds = [](const timeval &tv) { return tv.tv_sec + tv.tv_usec * 1e-6; };
    return to_seconds(usage.ru_utime) + to_seconds(usage.ru_stime);
#endif
}

// New-style thread pool.
struct NewThreadPool {
    static const char *name() { return "NewThreadPool"; }

    using ThreadPool = WorkerPool;

    // Not really executors, but matches the old style.
    struct Executors {
        TaskPool tasks;
        WorkerPool::BindTasks binder;

        Executors(ThreadPool &threads, std::size_t num_producers, std::size_t queue_capacity)
                : tasks(num_producers, queue_capacity), binder(threads, tasks) {}

        AsyncExecutor get(std::size_t idx) { return AsyncExecutor(tasks, idx); }
    };
};

enum class ProducerMode {
    Continuous,  ///< Continuous production of tasks.
    Burst,       ///< Bursts of jobs where the total job time matches the gaps between bursts.
    BurstHalf,   ///< Bursts of jobs where the total job time is half of the gaps between bursts.
};

template <typename Executor>
static void producer_thread(
        std::atomic_bool &finished,
        ProducerMode mode,
        Executor &&executor,  // NOLINT(cppcoreguidelines-missing-std-forward): universal reference
        std::size_t seed,
        std::atomic_size_t &counter) {
    // Randomly pick how long each task takes.
    std::minstd_rand rng(static_cast<std::minstd_rand::result_type>(seed));
    std::uniform_int_distribution<> time_dist(1, 100);

    // How many tasks to push per burst.
    constexpr std::size_t tasks_per_burst = 100;
    // avg 50us/task * 100tasks = 5000us to match in/out.
    std::chrono::microseconds burst_sleep(5'000);
    if (mode == ProducerMode::BurstHalf) {
        burst_sleep *= 2;
    }

    std::size_t pushed_tasks = 0;
    while (!finished.load(std::memory_order_relaxed)) {
        const std::chrono::microseconds task_time(time_dist(rng));
        executor.send([&counter, task_time] {
            busy_task(task_time);
            counter.fetch_add(1, std::memory_order_relaxed);
        });

        if (++pushed_tasks >= tasks_per_burst) {
            pushed_tasks = 0;
            if (mode != ProducerMode::Continuous) {
                std::this_thread::sleep_for(burst_sleep);
            }
        }
    }
};

DEFINE_TEMPLATE_TEST("Benchmarking", NewThreadPool) {
    const auto producer_mode =
            GENERATE(ProducerMode::Continuous, ProducerMode::Burst, ProducerMode::BurstHalf);
    const std::size_t num_workers = GENERATE(1, 2, 4, 8, 16);
    const std::size_t queue_capacity = 1'000;
    const auto run_for = std::chrono::seconds(2);
    const std::size_t max_threads = std::thread::hardware_concurrency();

    if (num_workers > max_threads) {
        CATCH_SKIP("Skipping benchmark for workers("
                   << num_workers << ") due to not enough CPU cores(" << max_threads << ")");
    }

    // Create the worker pool.
    typename TestType::ThreadPool thread_pool(num_workers, "test");

    // Typically we have a small number of producers vs a large number of workers.
    for (std::size_t num_producers : {1, 2, 4}) {
        std::atomic_size_t counter = 0;
        std::atomic_bool finished = false;  // TODO: remove and use jthread's stop_token
        const auto rusage_before = cpu_rusage();

        // Create the executors/task pool.
        typename TestType::Executors executors(thread_pool, num_producers, queue_capacity);

        // Make some infinitely generating producers.
        std::vector<dorado::utils::jthread> threads(num_producers);
        for (std::size_t idx = 0; idx < num_producers; idx++) {
            threads[idx] = dorado::utils::jthread(
                    [&finished, producer_mode, &counter, &executors, idx]() mutable {
                        producer_thread(finished, producer_mode, executors.get(idx), idx, counter);
                    });
        }

        // See how many we can process in a fixed amount of time.
        counter.store(0, std::memory_order_relaxed);
        std::this_thread::sleep_for(run_for);
        const std::size_t processed = counter.load(std::memory_order_relaxed);

        // Join the threads.
        finished.store(true, std::memory_order_relaxed);
        threads.clear();

        // Determine how much CPU we consumed during that run.
        const auto rusage_after = cpu_rusage();
        const auto cpu_used = rusage_after - rusage_before;

        DORADO_LOG(info, "[SPEED] [{}] mode={}, workers={}, producers={}: {} ({} CPU)",
                   TestType::name(), fmt::underlying(producer_mode), num_workers, num_producers,
                   processed, cpu_used);
    }
}

#endif

}  // namespace
