#include "utils/concurrency/detail/priority_task_queue.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#define CUT_TAG "[dorado::utils::concurrency::detail::PriorityTaskQueue]"
#define DEFINE_TEST(name) CATCH_TEST_CASE(CUT_TAG " " name, CUT_TAG)
#define DEFINE_CATCH_SCENARIO(name) CATCH_SCENARIO(CUT_TAG " " name, CUT_TAG)

namespace dorado::utils::concurrency::detail::priority_task_queue_test {

DEFINE_TEST("constructor does not throw") { CATCH_REQUIRE_NOTHROW(PriorityTaskQueue{}); }

DEFINE_TEST("size() after TaskQueue(high)::push") {
    PriorityTaskQueue cut{};
    auto& task_queue = cut.create_task_queue(TaskPriority::high);

    task_queue.push([] {});

    CATCH_CHECK(cut.size() == 1);
    CATCH_CHECK(cut.size(TaskPriority::high) == 1);
    CATCH_CHECK(cut.size(TaskPriority::normal) == 0);
}

DEFINE_TEST("size() after TaskQueue(normal)::push") {
    PriorityTaskQueue cut{};
    auto& task_queue = cut.create_task_queue(TaskPriority::normal);

    task_queue.push([] {});

    CATCH_CHECK(cut.size() == 1);
    CATCH_CHECK(cut.size(TaskPriority::high) == 0);
    CATCH_CHECK(cut.size(TaskPriority::normal) == 1);
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

    CATCH_CHECK(cut.size() == 9);
    CATCH_CHECK(cut.size(TaskPriority::high) == 6);
    CATCH_CHECK(cut.size(TaskPriority::normal) == 3);
}

DEFINE_CATCH_SCENARIO("prioritised pushing and popping with 2 high queues and one normal queue") {
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
        CATCH_CHECK(waiting_task.priority == priority);
        waiting_task.task();
        CATCH_CHECK(task_id == expected_task_id);
    };

    CATCH_GIVEN("2 tasks pushed to normal queue") {
        normal_queue_1.push(create_task("n1"));
        normal_queue_1.push(create_task("n2"));

        CATCH_THEN("pop() returns first normal") {
            check_task(cut.pop(), TaskPriority::normal, "n1");
        }
        CATCH_THEN("pop(normal) returns first normal") {
            check_task(cut.pop(TaskPriority::normal), TaskPriority::normal, "n1");

            CATCH_AND_THEN("pop() returns second normal") {
                check_task(cut.pop(TaskPriority::normal), TaskPriority::normal, "n2");
            }
        }
    }
    CATCH_GIVEN("1 tasks pushed to normal queue") {
        normal_queue_1.push(create_task("n1"));
        normal_queue_1.push(create_task("n2"));
        CATCH_AND_GIVEN("2 tasks pushed to first high_prio queue then second high prio queue") {
            high_queue_1.push(create_task("h1.1"));
            high_queue_1.push(create_task("h1.2"));
            high_queue_2.push(create_task("h2.1"));
            high_queue_2.push(create_task("h2.2"));

            CATCH_AND_GIVEN("1 further task to first high_prio queue") {
                CATCH_THEN("all tasks popped in order of cycling queues") {
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

DEFINE_CATCH_SCENARIO("Popping tasks") {
    CATCH_GIVEN("A high and a normal priority queue") {
        PriorityTaskQueue queue;
        auto& normal_prio_queue = queue.create_task_queue(TaskPriority::normal);
        auto& high_prio_queue = queue.create_task_queue(TaskPriority::high);

        CATCH_WHEN("A task is pushed to each queue") {
            // Check both orderings
            const auto normal_first = GENERATE(true, false);
            CATCH_CAPTURE(normal_first);
            const auto push_order = normal_first ? std::pair(&normal_prio_queue, &high_prio_queue)
                                                 : std::pair(&high_prio_queue, &normal_prio_queue);
            push_order.first->push([] {});
            push_order.second->push([] {});

            CATCH_THEN("Queue sizes match") {
                CATCH_CHECK(queue.size() == 2);
                CATCH_CHECK(queue.size(TaskPriority::high) == 1);
                CATCH_CHECK(queue.size(TaskPriority::normal) == 1);
            }

            CATCH_THEN("Popping explicit priorities match their priority") {
                // Check both orderings
                const auto high_first = GENERATE(false, true);
                const auto pop_order =
                        high_first ? std::pair(TaskPriority::high, TaskPriority::normal)
                                   : std::pair(TaskPriority::normal, TaskPriority::high);
                CATCH_CAPTURE(pop_order.first, pop_order.second);

                CATCH_CHECK(queue.pop(pop_order.first).priority == pop_order.first);
                CATCH_CHECK(queue.size() == 1);
                CATCH_CHECK(queue.size(pop_order.first) == 0);
                CATCH_CHECK(queue.size(pop_order.second) == 1);

                CATCH_CHECK(queue.pop(pop_order.second).priority == pop_order.second);
                CATCH_CHECK(queue.size() == 0);
                CATCH_CHECK(queue.size(pop_order.first) == 0);
                CATCH_CHECK(queue.size(pop_order.second) == 0);
            }

            CATCH_THEN("Popping 1 explicit priority and 1 arbitrary matches priorities") {
                // Check both orderings
                const auto priority = GENERATE(TaskPriority::normal, TaskPriority::high);
                CATCH_CAPTURE(priority);

                CATCH_CHECK(queue.pop(priority).priority == priority);
                CATCH_CHECK(queue.size() == 1);
                CATCH_CHECK(queue.size(priority) == 0);

                CATCH_CHECK(queue.pop().priority != priority);
                CATCH_CHECK(queue.size() == 0);
            }

            CATCH_THEN("Popping arbitrary tasks don't match each other") {
                const auto first_priority = queue.pop().priority;
                CATCH_CHECK(queue.size() == 1);
                CATCH_CHECK(queue.size(first_priority) == 0);

                CATCH_CHECK(queue.pop().priority != first_priority);
                CATCH_CHECK(queue.size() == 0);
            }
        }
    }
}

}  // namespace dorado::utils::concurrency::detail::priority_task_queue_test