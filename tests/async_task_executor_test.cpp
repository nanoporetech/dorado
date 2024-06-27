#include "utils/concurrency/async_task_executor.h"

#include "utils/PostCondition.h"
#include "utils/concurrency/synchronisation.h"

#include <catch2/catch.hpp>

#include <utility>

#define CUT_TAG "[dorado::utils::concurrency::AsyncTaskExecutor]"
#define DEFINE_TEST(name) TEST_CASE(CUT_TAG " " name, CUT_TAG)

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

}  // namespace dorado::utils::concurrency::async_task_executor