#include "utils/concurrency/async_task_executor.h"

#include "utils/PostCondition.h"
#include "utils/concurrency/synchronisation.h"

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <memory>
#include <thread>
#include <utility>

#define CUT_TAG "[dorado::utils::concurrency::AsyncTaskExecutor]"
#define DEFINE_TEST(name) CATCH_TEST_CASE(CUT_TAG " " name, CUT_TAG)
#define DEFINE_CATCH_SCENARIO(name) CATCH_SCENARIO(CUT_TAG " " name, CUT_TAG)

using namespace std::chrono_literals;

namespace dorado::utils::concurrency::async_task_executor {

namespace {

constexpr auto TIMEOUT{10s};
constexpr std::size_t MAX_QUEUE_SIZE{100};

}  // namespace

DEFINE_TEST("AsyncTaskExecutor constructor with valid thread pool does not throw") {
    MultiQueueThreadPool pool{1};
    CATCH_REQUIRE_NOTHROW(AsyncTaskExecutor(pool, TaskPriority::normal, MAX_QUEUE_SIZE));
}

DEFINE_TEST("AsyncTaskExecutor::send() does not throw") {
    MultiQueueThreadPool pool{2};
    AsyncTaskExecutor cut(pool, TaskPriority::normal, MAX_QUEUE_SIZE);

    CATCH_REQUIRE_NOTHROW(cut.send([] {}));
}

DEFINE_TEST("AsyncTaskExecutor::send() invokes the task") {
    MultiQueueThreadPool pool{2};
    AsyncTaskExecutor cut(pool, TaskPriority::normal, MAX_QUEUE_SIZE);
    Flag invoked{};

    cut.send([&invoked] { invoked.signal(); });

    CATCH_REQUIRE(invoked.wait_for(TIMEOUT));
}

DEFINE_TEST("AsyncTaskExecutor::send() with non-copyable task invokes the task") {
    MultiQueueThreadPool pool{2};
    AsyncTaskExecutor cut(pool, TaskPriority::normal, MAX_QUEUE_SIZE);

    Flag invoked{};
    struct Signaller {
        Signaller(Flag& flag) : m_flag(flag) {}
        void signal() { m_flag.signal(); }

        Flag& m_flag;
    };
    auto non_copyable_signaller = std::make_unique<Signaller>(invoked);

    cut.send([signaller = std::move(non_copyable_signaller)] { signaller->signal(); });

    CATCH_REQUIRE(invoked.wait_for(TIMEOUT));
}

DEFINE_CATCH_SCENARIO("AsyncTaskExecutor created with pool of 2 threads") {
    MultiQueueThreadPool pool{2};
    AsyncTaskExecutor cut(pool, TaskPriority::normal, MAX_QUEUE_SIZE);

    CATCH_GIVEN("2 tasks are running") {
        std::vector<std::unique_ptr<Flag>> task_release_flags{};
        task_release_flags.reserve(3);
        task_release_flags.emplace_back(std::make_unique<Flag>());
        task_release_flags.emplace_back(std::make_unique<Flag>());
        task_release_flags.emplace_back(std::make_unique<Flag>());

        std::vector<std::unique_ptr<Flag>> task_started_flags{};
        task_started_flags.reserve(3);
        task_started_flags.emplace_back(std::make_unique<Flag>());
        task_started_flags.emplace_back(std::make_unique<Flag>());
        task_started_flags.emplace_back(std::make_unique<Flag>());

        auto create_task = [&task_release_flags, &task_started_flags](std::size_t index) {
            return [&task_release_flags, &task_started_flags, index] {
                task_started_flags[index]->signal();
                task_release_flags[index]->wait();
            };
        };

        cut.send(create_task(0));
        cut.send(create_task(1));
        auto release_all_tasks = [&task_release_flags] {
            for (auto& release_flag : task_release_flags) {
                release_flag->signal();
            }
        };
        auto complete_all_tasks_on_exit = PostCondition([&release_all_tasks, &cut] {
            release_all_tasks();
            // Need to also ensure flushed before allowing the flags used by the tasks running
            // in the thread pool to go out of scope.
            cut.flush();
        });

        // it's required that both tasks have started to proceed
        CATCH_REQUIRE(task_started_flags[0]->wait_for(TIMEOUT));
        CATCH_REQUIRE(task_started_flags[1]->wait_for(TIMEOUT));

        CATCH_AND_GIVEN("third task is sent") {
            std::thread third_task_thread([&cut, &create_task] { cut.send(create_task(2)); });
            auto join_third_task_thread_on_exit =
                    PostCondition([&release_all_tasks, &third_task_thread] {
                        release_all_tasks();
                        if (third_task_thread.joinable()) {
                            third_task_thread.join();
                        }
                    });

            CATCH_THEN("third task is not invoked") {
                CATCH_CHECK_FALSE(task_started_flags[2]->wait_for(200ms));
            }

            CATCH_WHEN("first task completes") {
                task_release_flags[0]->signal();

                CATCH_THEN("third task is invoked") {
                    CATCH_CHECK(task_started_flags[2]->wait_for(TIMEOUT));
                }
            }
        }

        CATCH_WHEN("flush is called") {
            Flag flush_completed{};
            std::thread flushing_thread([&cut, &flush_completed] {
                cut.flush();
                flush_completed.signal();
            });
            auto join_flushing_thread_on_exit =
                    PostCondition([&release_all_tasks, &flushing_thread] {
                        release_all_tasks();
                        if (flushing_thread.joinable()) {
                            flushing_thread.join();
                        }
                    });

            CATCH_THEN("flush is blocked") { CATCH_CHECK_FALSE(flush_completed.wait_for(200ms)); }

            CATCH_AND_WHEN("one tasks is completed") {
                task_release_flags[0]->signal();

                CATCH_THEN("flush is still blocked") {
                    CATCH_CHECK_FALSE(flush_completed.wait_for(200ms));
                }
            }

            CATCH_AND_WHEN("all tasks are completed") {
                release_all_tasks();
                CATCH_THEN("flush is unblocked") { CATCH_CHECK(flush_completed.wait_for(TIMEOUT)); }
            }
        }
    }
}

DEFINE_TEST("AsyncTaskExecutor destructor blocks till all tasks completed") {
    MultiQueueThreadPool thread_pool{2};

    constexpr std::size_t NUM_TASKS{3};
    std::atomic_size_t num_completed_tasks{0};
    std::unique_ptr<std::thread> producer_thread{};
    std::unique_ptr<std::thread> third_task_thread{};
    auto join_producer_threads_on_exit = PostCondition([&producer_thread, &third_task_thread] {
        if (producer_thread && producer_thread->joinable()) {
            producer_thread->join();
        }
        if (third_task_thread && third_task_thread->joinable()) {
            third_task_thread->join();
        }
    });
    Flag unblock_tasks{};
    {
        AsyncTaskExecutor cut(thread_pool, TaskPriority::normal, MAX_QUEUE_SIZE);
        Latch two_tasks_running{2};
        producer_thread = std::make_unique<std::thread>(
                [&cut, &num_completed_tasks, &unblock_tasks, &two_tasks_running] {
                    for (std::size_t i{0}; i < 2; ++i) {
                        cut.send([&num_completed_tasks, &unblock_tasks, &two_tasks_running] {
                            two_tasks_running.count_down();
                            unblock_tasks.wait();
                            ++num_completed_tasks;
                        });
                    }
                });
        // When we know both pool threads are busy so post another task to that will have
        // to wait to be started.
        two_tasks_running.wait();
        third_task_thread = cut.send_async([&unblock_tasks, &num_completed_tasks] {
            unblock_tasks.wait();
            ++num_completed_tasks;
        });

        // 2 tasks running + 1 task enqueued, so now unblock the tasks and immediately let
        // the AsyncTaskExecutor go out of scope and check that all tasks completed
        unblock_tasks.signal();
    }
    CATCH_REQUIRE(num_completed_tasks == NUM_TASKS);
}

}  // namespace dorado::utils::concurrency::async_task_executor