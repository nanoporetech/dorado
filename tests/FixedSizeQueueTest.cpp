#include "utils/FixedSizeQueue.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <memory>

#define TEST_GROUP "FixedSizeQueue"
#define DEFINE_TEST(name) CATCH_TEST_CASE(TEST_GROUP " : " name, TEST_GROUP)

using dorado::utils::FixedSizeQueue;

DEFINE_TEST("smoke test") {
    const std::size_t capacity = GENERATE(1, 2);
    CATCH_CAPTURE(capacity);

    FixedSizeQueue<int> queue(capacity);
    CATCH_CHECK(queue.capacity() == capacity);
    CATCH_CHECK(queue.size() == 0);
    CATCH_CHECK_FALSE(queue.full());
    CATCH_CHECK(queue.empty());

    // Push an item.
    queue.push(123);
    CATCH_CHECK(queue.capacity() == capacity);
    CATCH_CHECK(queue.size() == 1);
    const bool is_full = capacity == 1;
    CATCH_CHECK(queue.full() == is_full);
    CATCH_CHECK_FALSE(queue.empty());

    // Pop the item.
    const int value = queue.pop();
    CATCH_CHECK(value == 123);
    CATCH_CHECK(queue.capacity() == capacity);
    CATCH_CHECK(queue.size() == 0);
    CATCH_CHECK_FALSE(queue.full());
    CATCH_CHECK(queue.empty());
}

DEFINE_TEST("destructors run") {
    auto counter = std::make_shared<int>();
    CATCH_CHECK(counter.use_count() == 1);

    {
        FixedSizeQueue<std::shared_ptr<int>> queue(10);

        // Adding items should increase the reference count.
        queue.push(counter);
        CATCH_CHECK(counter.use_count() == 2);
        queue.push(counter);
        CATCH_CHECK(counter.use_count() == 3);

        // A copy of a popped item shouldn't remain in the queue.
        (void)queue.pop();
        CATCH_CHECK(counter.use_count() == 2);

        // Destroying them should reduce the count.
        queue.clear();
        CATCH_CHECK(counter.use_count() == 1);

        // Adding more items again.
        queue.push(counter);
        CATCH_CHECK(counter.use_count() == 2);
        queue.push(counter);
        CATCH_CHECK(counter.use_count() == 3);
    }

    // Destruction of the queue should run item destructors.
    CATCH_CHECK(counter.use_count() == 1);
}

DEFINE_TEST("movable only types") {
    FixedSizeQueue<std::unique_ptr<int>> queue(10);

    queue.push(std::make_unique<int>(123));
    queue.push(std::make_unique<int>(456));

    auto item = queue.pop();
    CATCH_REQUIRE(item != nullptr);
    CATCH_CHECK(*item == 123);

    item = queue.pop();
    CATCH_REQUIRE(item != nullptr);
    CATCH_CHECK(*item == 456);
}

DEFINE_TEST("lots of items retain order") {
    const std::size_t capacity = 10;
    FixedSizeQueue<int> queue(capacity);

    // Push and pop items, making sure that the're in order.
    const struct {
        std::size_t num_pushes;
        std::size_t num_pops;
    } num_push_pops[] = {
            {5, 3},  // +2 -> 2
            {6, 1},  // +5 -> 7
            {1, 7},  // -6 -> 1
            {9, 8},  // +1 -> 2
            {8, 5},  // +3 -> 5
            {0, 4},  // -4 -> 1
    };

    int push_counter = 0;
    int pop_counter = 0;
    std::size_t current_size = 0;
    for (auto && [num_pushes, num_pops] : num_push_pops) {
        CATCH_CAPTURE(num_pushes, num_pops);

        CATCH_REQUIRE(current_size + num_pushes <= capacity);
        for (std::size_t i = 0; i < num_pushes; ++i) {
            queue.push(push_counter);
            push_counter++;
            current_size++;
            CATCH_CHECK(queue.size() == current_size);
        }

        CATCH_REQUIRE(current_size >= num_pops);
        for (std::size_t i = 0; i < num_pops; ++i) {
            CATCH_CHECK(queue.pop() == pop_counter);
            pop_counter++;
            current_size--;
            CATCH_CHECK(queue.size() == current_size);
        }
    }
}

DEFINE_TEST("wrap behaviour") {
    for (std::size_t capacity = 1; capacity < 10; capacity++) {
        CATCH_CAPTURE(capacity);
        FixedSizeQueue<std::size_t> queue(capacity);

        // Do it multiple times to test wrap behaviour.
        for (std::size_t repeat = 0; repeat < 3; repeat++) {
            // Push items in.
            for (std::size_t i = 0; i < capacity; i++) {
                CATCH_CHECK(queue.size() == i);
                CATCH_CHECK_FALSE(queue.full());
                queue.push(i);
            }
            CATCH_CHECK(queue.size() == capacity);
            CATCH_CHECK(queue.full());

            // Pull items out.
            for (std::size_t i = 0; i < capacity; i++) {
                CATCH_CHECK(queue.size() == capacity - i);
                CATCH_CHECK_FALSE(queue.empty());
                const std::size_t val = queue.pop();
                CATCH_CHECK(val == i);
                CATCH_CHECK_FALSE(queue.full());
            }
            CATCH_CHECK(queue.size() == 0);
            CATCH_CHECK(queue.empty());
        }
    }
}

DEFINE_TEST("alignment check") {
    // GCC's UBSan has issues with creating overly aligned structs on the stack, so
    // use a smaller alignment in that case.
#if defined(__GNUC__) && !defined(__clang__)
    static constexpr std::size_t TestAlignment = alignof(std::max_align_t);
#else
    static constexpr std::size_t TestAlignment = 128;
#endif

    struct TestType {
        alignas(TestAlignment) std::uint8_t i;

        ~TestType() {
            const auto addr = reinterpret_cast<std::uintptr_t>(this);
            const auto alignment = addr % TestAlignment;
            CATCH_CHECK(alignment == 0);
        }
    };
    static_assert(sizeof(TestType) == TestAlignment);
    static_assert(alignof(TestType) == TestAlignment);

    FixedSizeQueue<TestType> queue(10);

    for (std::uint8_t i = 0; i < 3; i++) {
        queue.push(TestType{i});
    }
}
