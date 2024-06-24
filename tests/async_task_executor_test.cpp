#include "utils/concurrency/async_task_executor.h"

#include "utils/concurrency/synchronisation.h"

#include <catch2/catch.hpp>

#define CUT_TAG "[dorado::utils::concurrency::AsyncTaskExecutor]"
#define DEFINE_TEST(name) TEST_CASE(CUT_TAG " " name, CUT_TAG)

using namespace std::chrono_literals;

namespace dorado::utils::concurrency::async_task_executor {

namespace {
constexpr auto TIMEOUT{5s};
}

DEFINE_TEST("Constructor with 1 thread does not throw") { REQUIRE_NOTHROW(AsyncTaskExecutor(1)); }

DEFINE_TEST("send() with task does not thread") {
    AsyncTaskExecutor cut{1};

    REQUIRE_NOTHROW(cut.send([] {}));
}

DEFINE_TEST("send() with task invokes the task") {
    AsyncTaskExecutor cut{1};

    Flag invoked{};

    cut.send([&invoked] { invoked.signal(); });

    REQUIRE(invoked.wait_for(TIMEOUT));
}

DEFINE_TEST("send() invokes task on separate thread") {
    AsyncTaskExecutor cut{1};

    Flag thread_id_assigned{};

    auto invocation_thread{std::this_thread::get_id()};

    cut.send([&thread_id_assigned, &invocation_thread] {
        invocation_thread = std::this_thread::get_id();
        thread_id_assigned.signal();
    });

    CHECK(thread_id_assigned.wait_for(TIMEOUT));
    REQUIRE(invocation_thread != std::this_thread::get_id());
}

}  // namespace dorado::utils::concurrency::async_task_executor