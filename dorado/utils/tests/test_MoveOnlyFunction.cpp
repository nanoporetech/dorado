#include "utils/MoveOnlyFunction.h"

#include <catch2/catch_test_macros.hpp>

#include <functional>
#include <utility>

#define CUT_TAG "[MoveOnlyFunction]"
#define DEFINE_TEST(name) CATCH_TEST_CASE(CUT_TAG " " name, CUT_TAG)

using namespace dorado::utils;

namespace {

DEFINE_TEST("Empty behaviour") {
    MoveOnlyFunction<void()> empty;
    CATCH_CHECK_FALSE(empty);
    CATCH_CHECK_THROWS_AS(empty(), std::bad_function_call);

    MoveOnlyFunction<void()> moved_ctor = std::move(empty);
    CATCH_CHECK_FALSE(empty);
    CATCH_CHECK_FALSE(moved_ctor);
    CATCH_CHECK_THROWS_AS(moved_ctor(), std::bad_function_call);

    MoveOnlyFunction<void()> moved_assigned;
    moved_assigned = std::move(moved_ctor);
    CATCH_CHECK_FALSE(moved_ctor);
    CATCH_CHECK_FALSE(moved_assigned);
    CATCH_CHECK_THROWS_AS(moved_assigned(), std::bad_function_call);
}

DEFINE_TEST("Callable is called once") {
    int counter = 0;

    MoveOnlyFunction<int(int)> add_to_counter = [&](int x) { return counter += x; };
    CATCH_CHECK(add_to_counter(2) == 2);
    CATCH_CHECK(counter == 2);

    MoveOnlyFunction<int(int)> moved_ctor = std::move(add_to_counter);
    CATCH_CHECK_FALSE(add_to_counter);
    CATCH_CHECK(moved_ctor(3) == 5);
    CATCH_CHECK(counter == 5);

    MoveOnlyFunction<int(int)> moved_assigned;
    moved_assigned = std::move(moved_ctor);
    CATCH_CHECK_FALSE(moved_ctor);
    CATCH_CHECK(moved_assigned(4) == 9);
    CATCH_CHECK(counter == 9);
}

DEFINE_TEST("Move only lambda") {
    auto counter_ptr = std::make_unique<int>();
    const int &counter = *counter_ptr;

    auto lambda = [ptr = std::move(counter_ptr)] { *ptr += 1; };
    using Lambda = std::remove_reference_t<decltype(lambda)>;
    static_assert(!std::is_copy_constructible_v<Lambda> && !std::is_copy_assignable_v<Lambda>);

    MoveOnlyFunction<void()> functor = std::move(lambda);
    CATCH_CHECK(counter == 0);
    functor();
    CATCH_CHECK(counter == 1);
    functor();
    CATCH_CHECK(counter == 2);
}

static int add_one_func(int x) { return x + 1; };

DEFINE_TEST("Constructible from function pointer") {
    using FnType = int(int);
    FnType *const add_one_ptr = add_one_func;

    MoveOnlyFunction<FnType> functor = add_one_ptr;
    CATCH_CHECK(functor(2) == 3);
    functor = add_one_ptr;
    CATCH_CHECK(functor(3) == 4);

    // nullptr is a valid pointer
    functor = nullptr;
    CATCH_CHECK_FALSE(functor);
    functor = (FnType *)nullptr;
    CATCH_CHECK_FALSE(functor);

    MoveOnlyFunction<FnType> from_func = add_one_func;
    CATCH_CHECK(from_func(4) == 5);
    from_func = add_one_func;
    CATCH_CHECK(from_func(5) == 6);
}

DEFINE_TEST("Invocable with refs") {
    auto lambda = [](int &x) { x++; };
    MoveOnlyFunction<void(int &)> functor = lambda;

    int counter = 0;
    functor(counter);
    CATCH_CHECK(counter == 1);
}

DEFINE_TEST("const correctness") {
    static_assert(std::is_invocable_v<MoveOnlyFunction<void(void)>>);
    static_assert(!std::is_invocable_v<const MoveOnlyFunction<void()>>);
    static_assert(std::is_invocable_v<MoveOnlyFunction<void() const>>);
    static_assert(std::is_invocable_v<const MoveOnlyFunction<void() const>>);

    MoveOnlyFunction<void(int &) const> add_one = [](int &x) { x++; };

    int counter = 0;
    add_one(counter);
    CATCH_CHECK(counter == 1);
    std::as_const(add_one)(counter);
    CATCH_CHECK(counter == 2);
}

DEFINE_TEST("All storage types work") {
    using AddOneFunc = MoveOnlyFunction<void(int &) const>;

    // Utils to make each type of storage.
    const auto make_func_ptr = [] {
        auto *func_ptr = +[](int &counter) { counter++; };
        return func_ptr;
    };
    const auto make_small_functor = [] {
        const auto small_lambda = [](int &counter) { counter++; };
        static_assert(detail::is_small_v<decltype(small_lambda)>);
        return small_lambda;
    };
    const auto make_big_functor = [] {
        struct Padding {
            Padding() {}  // trick the compiler into thinking that this does something
            char pad[128];
        };
        const auto big_lambda = [padding = Padding()](int &counter) { counter++; };
        static_assert(!detail::is_small_v<decltype(big_lambda)>);
        return big_lambda;
    };

    // Sanity check that they all do the same thing.
    int counter = 0;
    const auto run_test = [&counter](const AddOneFunc &add_one) {
        const int before = counter;
        add_one(counter);
        CATCH_CHECK(counter == before + 1);
    };
    run_test(make_func_ptr());
    run_test(make_small_functor());
    run_test(make_big_functor());

    // Check assignment to/from each type.
    const auto assign_from = [&](auto from) {
        {
            AddOneFunc functor = make_func_ptr();
            run_test(functor);
            functor = from;
            run_test(functor);
        }
        {
            AddOneFunc functor = make_small_functor();
            run_test(functor);
            functor = from;
            run_test(functor);
        }
        {
            AddOneFunc functor = make_big_functor();
            run_test(functor);
            functor = from;
            run_test(functor);
        }
    };
    assign_from(make_func_ptr());
    assign_from(make_small_functor());
    assign_from(make_big_functor());
}

}  // namespace
