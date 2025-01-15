#include "utils/concurrency/detail/priority_task_queue.h"

#include <catch2/catch_all.hpp>

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

    auto check_task = [&task_id](const WaitingTask& waiting_task, TaskPriority priority,
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

DEFINE_SCENARIO("Popping tasks") {
    GIVEN("A high and a normal priority queue") {
        PriorityTaskQueue queue;
        auto& normal_prio_queue = queue.create_task_queue(TaskPriority::normal);
        auto& high_prio_queue = queue.create_task_queue(TaskPriority::high);

        WHEN("A task is pushed to each queue") {
            // Check both orderings
            const auto normal_first = GENERATE(true, false);
            CAPTURE(normal_first);
            const auto push_order = normal_first ? std::pair(&normal_prio_queue, &high_prio_queue)
                                                 : std::pair(&high_prio_queue, &normal_prio_queue);
            push_order.first->push([] {});
            push_order.second->push([] {});

            THEN("Queue sizes match") {
                CHECK(queue.size() == 2);
                CHECK(queue.size(TaskPriority::high) == 1);
                CHECK(queue.size(TaskPriority::normal) == 1);
            }

            THEN("Popping explicit priorities match their priority") {
                // Check both orderings
                const auto pop_order =
                        GENERATE(std::pair(TaskPriority::normal, TaskPriority::high),
                                 std::pair(TaskPriority::high, TaskPriority::normal));
                CAPTURE(pop_order.first, pop_order.second);

                CHECK(queue.pop(pop_order.first).priority == pop_order.first);
                CHECK(queue.size() == 1);
                CHECK(queue.size(pop_order.first) == 0);
                CHECK(queue.size(pop_order.second) == 1);

                CHECK(queue.pop(pop_order.second).priority == pop_order.second);
                CHECK(queue.size() == 0);
                CHECK(queue.size(pop_order.first) == 0);
                CHECK(queue.size(pop_order.second) == 0);
            }

            THEN("Popping 1 explicit priority and 1 arbitrary matches priorities") {
                // Check both orderings
                const auto priority = GENERATE(TaskPriority::normal, TaskPriority::high);
                CAPTURE(priority);

                CHECK(queue.pop(priority).priority == priority);
                CHECK(queue.size() == 1);
                CHECK(queue.size(priority) == 0);

                CHECK(queue.pop().priority != priority);
                CHECK(queue.size() == 0);
            }

            THEN("Popping arbitrary tasks don't match each other") {
                const auto first_priority = queue.pop().priority;
                CHECK(queue.size() == 1);
                CHECK(queue.size(first_priority) == 0);

                CHECK(queue.pop().priority != first_priority);
                CHECK(queue.size() == 0);
            }
        }
    }
}

}  // namespace dorado::utils::concurrency::detail::priority_task_queue_test