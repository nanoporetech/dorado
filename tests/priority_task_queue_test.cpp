#include "utils/concurrency/detail/priority_task_queue.h"

#include <catch2/catch.hpp>

#define CUT_TAG "[dorado::utils::concurrency::detail::PriorityTaskQueue]"
#define DEFINE_TEST(name) TEST_CASE(CUT_TAG " " name, CUT_TAG)
#define DEFINE_SCENARIO(name) SCENARIO(CUT_TAG " " name, CUT_TAG)

namespace dorado::utils::concurrency::detail::priority_task_queue_test {

DEFINE_TEST("constructor does not throw") { REQUIRE_NOTHROW(PriorityTaskQueue{}); }

DEFINE_TEST("push() does not throw") {
    PriorityTaskQueue cut{};

    REQUIRE_NOTHROW(cut.push(std::make_shared<WaitingTask>([] {}, TaskPriority::normal)));
}

DEFINE_TEST("size() after push(high)") {
    PriorityTaskQueue cut{};

    cut.push(std::make_shared<WaitingTask>([] {}, TaskPriority::high));

    CHECK(cut.size() == 1);
    CHECK(cut.size(TaskPriority::high) == 1);
    CHECK(cut.size(TaskPriority::normal) == 0);
}

DEFINE_TEST("size() after push(normal)") {
    PriorityTaskQueue cut{};

    cut.push(std::make_shared<WaitingTask>([] {}, TaskPriority::normal));

    CHECK(cut.size() == 1);
    CHECK(cut.size(TaskPriority::high) == 0);
    CHECK(cut.size(TaskPriority::normal) == 1);
}

DEFINE_TEST("size() after pushing 3 normal 2 high") {
    PriorityTaskQueue cut{};

    cut.push(std::make_shared<WaitingTask>([] {}, TaskPriority::normal));
    cut.push(std::make_shared<WaitingTask>([] {}, TaskPriority::high));
    cut.push(std::make_shared<WaitingTask>([] {}, TaskPriority::normal));
    cut.push(std::make_shared<WaitingTask>([] {}, TaskPriority::high));
    cut.push(std::make_shared<WaitingTask>([] {}, TaskPriority::normal));

    CHECK(cut.size() == 5);
    CHECK(cut.size(TaskPriority::high) == 2);
    CHECK(cut.size(TaskPriority::normal) == 3);
}

DEFINE_SCENARIO("prioritised pushing and popping") {
    PriorityTaskQueue cut{};
    std::vector<std::shared_ptr<WaitingTask>> normal{};
    std::vector<std::shared_ptr<WaitingTask>> high{};
    for (std::size_t index{0}; index < 3; ++index) {
        normal.push_back(std::make_shared<WaitingTask>([] {}, TaskPriority::normal));
        high.push_back(std::make_shared<WaitingTask>([] {}, TaskPriority::high));
    }

    auto check_sizes = [&cut](std::size_t num_normal, std::size_t num_high, std::size_t total) {
        CHECK(cut.size(TaskPriority::normal) == num_normal);
        CHECK(cut.size(TaskPriority::high) == num_high);
        CHECK(cut.size() == total);
    };

    GIVEN("2 normal tasks pushed") {
        cut.push(normal[0]);
        cut.push(normal[1]);

        THEN("pop() returns first normal") { CHECK(cut.pop().get() == normal[0].get()); }
        THEN("pop(normal) returns first normal") {
            CHECK(cut.pop(TaskPriority::normal).get() == normal[0].get());

            AND_THEN("pop() returns second normal") { CHECK(cut.pop().get() == normal[1].get()); }
        }

        WHEN("push(high)") {
            cut.push(high[0]);
            THEN("pop() returns first normal") { CHECK(cut.pop().get() == normal[0].get()); }
            THEN("pop(normal) returns first normal") {
                CHECK(cut.pop(TaskPriority::normal).get() == normal[0].get());

                AND_THEN("sizes correct") { check_sizes(1, 1, 2); }
            }
            THEN("pop(high) returns first high") {
                CHECK(cut.pop(TaskPriority::high).get() == high[0].get());

                AND_THEN("pop() returns first normal") {
                    CHECK(cut.pop().get() == normal[0].get());
                }

                AND_THEN("sizes correct") { check_sizes(2, 0, 2); }
            }

            AND_WHEN("push(high) push(high) push(normal)") {
                cut.push(high[1]);
                cut.push(high[2]);
                cut.push(normal[2]);

                THEN("sizes correct") { check_sizes(3, 3, 6); }

                THEN("pop(high) returns first high") {
                    CHECK(cut.pop(TaskPriority::high).get() == high[0].get());

                    AND_THEN("sizes correct") { check_sizes(3, 2, 5); }

                    AND_THEN("pop() pop() pop() returns first normal, second normal, second high") {
                        CHECK(cut.pop().get() == normal[0].get());
                        CHECK(cut.pop().get() == normal[1].get());
                        CHECK(cut.pop().get() == high[1].get());

                        AND_THEN("sizes correct") { check_sizes(1, 1, 2); }

                        AND_THEN("pop(normal) pop() returns third normal then third high") {
                            CHECK(cut.pop(TaskPriority::normal).get() == normal[2].get());
                            CHECK(cut.pop().get() == high[2].get());
                            CHECK(cut.empty());

                            AND_THEN("sizes correct") { check_sizes(0, 0, 0); }
                        }
                        AND_THEN("pop() pop() returns third high then third normal") {
                            CHECK(cut.pop().get() == high[2].get());
                            CHECK(cut.pop().get() == normal[2].get());
                            CHECK(cut.empty());

                            AND_THEN("sizes correct") { check_sizes(0, 0, 0); }
                        }
                    }
                }
            }
        }
    }
}
}  // namespace dorado::utils::concurrency::detail::priority_task_queue_test