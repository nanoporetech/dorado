#include "utils/concurrency/detail/priority_task_queue.h"

#include <catch2/catch.hpp>

#define CUT_TAG "[dorado::utils::concurrency::detail::PriorityTaskQueue]"
#define DEFINE_TEST(name) TEST_CASE(CUT_TAG " " name, CUT_TAG)
#define DEFINE_SCENARIO(name) SCENARIO(CUT_TAG " " name, CUT_TAG)

namespace dorado::utils::concurrency::detail::priority_task_queue_test {

DEFINE_TEST("constructor does not throw") { REQUIRE_NOTHROW(PriorityTaskQueue{}); }

DEFINE_TEST("size() after TaskQueue(high)::push") {
    PriorityTaskQueue cut{};
    auto& task_queue = cut.create_task_queue(TaskPriority::high);

    task_queue.push([] {});

    CHECK(cut.size() == 1);
    CHECK(cut.size(TaskPriority::high) == 1);
    CHECK(cut.size(TaskPriority::normal) == 0);
}

DEFINE_TEST("size() after TaskQueue(normal)::push") {
    PriorityTaskQueue cut{};
    auto& task_queue = cut.create_task_queue(TaskPriority::normal);

    task_queue.push([] {});

    CHECK(cut.size() == 1);
    CHECK(cut.size(TaskPriority::high) == 0);
    CHECK(cut.size(TaskPriority::normal) == 1);
}

DEFINE_TEST("size() after pushing to multiple tasks to multiple queues") {
    PriorityTaskQueue cut{};
    auto& normal_queue_1 = cut.create_task_queue(TaskPriority::normal);
    auto& normal_queue_2 = cut.create_task_queue(TaskPriority::normal);
    normal_queue_1.push([] {});
    normal_queue_2.push([] {});
    normal_queue_2.push([] {});

    auto& high_queue_1 = cut.create_task_queue(TaskPriority::high);
    auto& high_queue_2 = cut.create_task_queue(TaskPriority::high);
    auto& high_queue_3 = cut.create_task_queue(TaskPriority::high);
    high_queue_1.push([] {});
    high_queue_2.push([] {});
    high_queue_2.push([] {});
    high_queue_3.push([] {});
    high_queue_3.push([] {});
    high_queue_3.push([] {});

    CHECK(cut.size() == 9);
    CHECK(cut.size(TaskPriority::high) == 6);
    CHECK(cut.size(TaskPriority::normal) == 3);
}

DEFINE_SCENARIO("prioritised pushing and popping with 2 high queues and one normal queue") {
    PriorityTaskQueue cut{};
    auto& normal_queue_1 = cut.create_task_queue(TaskPriority::normal);

    auto& high_queue_1 = cut.create_task_queue(TaskPriority::high);
    auto& high_queue_2 = cut.create_task_queue(TaskPriority::high);

    std::string task_id{};
    auto create_task = [&task_id](const std::string& id) {
        return [&task_id, id] { task_id = id; };
    };

    auto check_task = [&task_id](WaitingTask waiting_task, TaskPriority priority,
                                 const std::string& expected_task_id) {
        CHECK(waiting_task.priority == priority);
        waiting_task.task();
        CHECK(task_id == expected_task_id);
    };

    GIVEN("2 tasks pushed to normal queue") {
        normal_queue_1.push(create_task("n1"));
        normal_queue_1.push(create_task("n2"));

        THEN("pop() returns first normal") { check_task(cut.pop(), TaskPriority::normal, "n1"); }
        THEN("pop(normal) returns first normal") {
            check_task(cut.pop(TaskPriority::normal), TaskPriority::normal, "n1");

            AND_THEN("pop() returns second normal") {
                check_task(cut.pop(TaskPriority::normal), TaskPriority::normal, "n2");
            }
        }
    }
    GIVEN("1 tasks pushed to normal queue") {
        normal_queue_1.push(create_task("n1"));
        normal_queue_1.push(create_task("n2"));
        AND_GIVEN("2 tasks pushed to first high_prio queue then second high prio queue") {
            high_queue_1.push(create_task("h1.1"));
            high_queue_1.push(create_task("h1.2"));
            high_queue_2.push(create_task("h2.1"));
            high_queue_2.push(create_task("h2.2"));

            AND_GIVEN("1 further task to first high_prio queue") {
                THEN("all tasks popped in order of cycling queues") {
                    check_task(cut.pop(), TaskPriority::normal, "n1");
                    check_task(cut.pop(), TaskPriority::high, "h1.1");
                    check_task(cut.pop(), TaskPriority::high, "h2.1");
                    check_task(cut.pop(), TaskPriority::normal, "n2");
                    check_task(cut.pop(), TaskPriority::high, "h1.2");
                    check_task(cut.pop(), TaskPriority::high, "h2.2");
                }
            }
        }
    }
}

//DEFINE_SCENARIO("prioritised pushing and popping") {
//    PriorityTaskQueue cut{};
//    std::vector<std::shared_ptr<WaitingTask>> normal{};
//    std::vector<std::shared_ptr<WaitingTask>> high{};
//    for (std::size_t index{0}; index < 3; ++index) {
//        normal.push_back(std::make_shared<WaitingTask>([] {}, TaskPriority::normal));
//        high.push_back(std::make_shared<WaitingTask>([] {}, TaskPriority::high));
//    }
//
//    auto check_sizes = [&cut](std::size_t num_normal, std::size_t num_high, std::size_t total) {
//        CHECK(cut.size(TaskPriority::normal) == num_normal);
//        CHECK(cut.size(TaskPriority::high) == num_high);
//        CHECK(cut.size() == total);
//    };
//
//    GIVEN("2 normal tasks pushed") {
//        cut.push(normal[0]);
//        cut.push(normal[1]);
//
//        THEN("pop() returns first normal") { CHECK(cut.pop().get() == normal[0].get()); }
//        THEN("pop(normal) returns first normal") {
//            CHECK(cut.pop(TaskPriority::normal).get() == normal[0].get());
//
//            AND_THEN("pop() returns second normal") { CHECK(cut.pop().get() == normal[1].get()); }
//        }
//
//        WHEN("push(high)") {
//            cut.push(high[0]);
//            THEN("pop() returns first normal") { CHECK(cut.pop().get() == normal[0].get()); }
//            THEN("pop(normal) returns first normal") {
//                CHECK(cut.pop(TaskPriority::normal).get() == normal[0].get());
//
//                AND_THEN("sizes correct") { check_sizes(1, 1, 2); }
//            }
//            THEN("pop(high) returns first high") {
//                CHECK(cut.pop(TaskPriority::high).get() == high[0].get());
//
//                AND_THEN("pop() returns first normal") {
//                    CHECK(cut.pop().get() == normal[0].get());
//                }
//
//                AND_THEN("sizes correct") { check_sizes(2, 0, 2); }
//            }
//
//            AND_WHEN("push(high) push(high) push(normal)") {
//                cut.push(high[1]);
//                cut.push(high[2]);
//                cut.push(normal[2]);
//
//                THEN("sizes correct") { check_sizes(3, 3, 6); }
//
//                THEN("pop(high) returns first high") {
//                    CHECK(cut.pop(TaskPriority::high).get() == high[0].get());
//
//                    AND_THEN("sizes correct") { check_sizes(3, 2, 5); }
//
//                    AND_THEN("pop() pop() pop() returns first normal, second normal, second high") {
//                        CHECK(cut.pop().get() == normal[0].get());
//                        CHECK(cut.pop().get() == normal[1].get());
//                        CHECK(cut.pop().get() == high[1].get());
//
//                        AND_THEN("sizes correct") { check_sizes(1, 1, 2); }
//
//                        AND_THEN("pop(normal) pop() returns third normal then third high") {
//                            CHECK(cut.pop(TaskPriority::normal).get() == normal[2].get());
//                            CHECK(cut.pop().get() == high[2].get());
//                            CHECK(cut.empty());
//
//                            AND_THEN("sizes correct") { check_sizes(0, 0, 0); }
//                        }
//                        AND_THEN("pop() pop() returns third high then third normal") {
//                            CHECK(cut.pop().get() == high[2].get());
//                            CHECK(cut.pop().get() == normal[2].get());
//                            CHECK(cut.empty());
//
//                            AND_THEN("sizes correct") { check_sizes(0, 0, 0); }
//                        }
//                    }
//                }
//            }
//        }
//    }
//}
}  // namespace dorado::utils::concurrency::detail::priority_task_queue_test