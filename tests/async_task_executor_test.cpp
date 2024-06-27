#include "utils/concurrency/async_task_executor.h"

#include "utils/PostCondition.h"
#include "utils/concurrency/synchronisation.h"

#include <catch2/catch.hpp>

#include <memory>
#include <thread>
#include <utility>

#define CUT_TAG "[dorado::utils::concurrency::AsyncTaskExecutor]"
#define DEFINE_TEST(name) TEST_CASE(CUT_TAG " " name, CUT_TAG)
#define DEFINE_SCENARIO(name) SCENARIO(CUT_TAG " " name, CUT_TAG)

using namespace std::chrono_literals;

namespace dorado::utils::concurrency::async_task_executor {

namespace {

constexpr auto TIMEOUT{10s};

}  // namespace

DEFINE_TEST("AsyncTaskExecutor constructor with valid thread pool does not throw") {
    REQUIRE_NOTHROW(AsyncTaskExecutor(std::make_shared<NoQueueThreadPool>(1)));
}

DEFINE_TEST("AsyncTaskExecutor::send() does not throw") {
    AsyncTaskExecutor cut(std::make_shared<NoQueueThreadPool>(2));

    REQUIRE_NOTHROW(cut.send([] {}));
}

DEFINE_TEST("AsyncTaskExecutor::send() invokes the task") {
    AsyncTaskExecutor cut(std::make_shared<NoQueueThreadPool>(2));
    Flag invoked{};

    cut.send([&invoked] { invoked.signal(); });

    REQUIRE(invoked.wait_for(TIMEOUT));
}

DEFINE_TEST("AsyncTaskExecutor::send() with non-copyable task invokes the task") {
    AsyncTaskExecutor cut(std::make_shared<NoQueueThreadPool>(2));

    Flag invoked{};
    struct Signaller {
        Signaller(Flag& flag) : m_flag(flag) {}
        void signal() { m_flag.signal(); }

        Flag& m_flag;
    };
    auto non_copyable_signaller = std::make_unique<Signaller>(invoked);

    cut.send([signaller = std::move(non_copyable_signaller)] { signaller->signal(); });

    REQUIRE(invoked.wait_for(TIMEOUT));
}

DEFINE_SCENARIO("AsyncTaskExecutor created with pool of 2 threads") {
    AsyncTaskExecutor cut(std::make_shared<NoQueueThreadPool>(2));

    GIVEN("2 tasks are running") {
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
        auto release_all_tasks_on_exit =
                PostCondition([&release_all_tasks] { release_all_tasks(); });

        // check they've both started and the worker threads are busy
        CHECK(task_started_flags[0]->wait_for(TIMEOUT));
        CHECK(task_started_flags[1]->wait_for(TIMEOUT));

        AND_GIVEN("third task is sent") {
            std::thread third_task_thread(
                    [&cut, third_task = create_task(2)] { cut.send(std::move(third_task)); });
            auto join_third_task_thread_on_exit =
                    PostCondition([&release_all_tasks, &third_task_thread] {
                        release_all_tasks();
                        if (third_task_thread.joinable()) {
                            third_task_thread.join();
                        }
                    });

            THEN("third task is not invoked") {
                CHECK_FALSE(task_started_flags[2]->wait_for(200ms));
            }

            WHEN("first task completes") {
                task_release_flags[0]->signal();

                THEN("third task is invoked") { CHECK(task_started_flags[2]->wait_for(TIMEOUT)); }
            }
        }

        WHEN("flush is called") {
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

            THEN("flush is blocked") { CHECK_FALSE(flush_completed.wait_for(200ms)); }

            AND_WHEN("one tasks is completed") {
                task_release_flags[0]->signal();

                THEN("flush is still blocked") { CHECK_FALSE(flush_completed.wait_for(200ms)); }
            }

            AND_WHEN("all tasks are completed") {
                release_all_tasks();
                THEN("flush is unblocked") { CHECK(flush_completed.wait_for(TIMEOUT)); }
            }
        }
    }
}

}  // namespace dorado::utils::concurrency::async_task_executor