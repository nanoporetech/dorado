#include "utils/FunctionRef.h"

#include <catch2/catch_test_macros.hpp>

#include <functional>
#include <utility>

#define CUT_TAG "[FunctionRef]"
#define DEFINE_TEST(name) CATCH_TEST_CASE(CUT_TAG " " name, CUT_TAG)

using namespace dorado::utils;

namespace {

template <typename Func, typename... Args>
decltype(auto) call_functor(FunctionRef<Func> function_ref, Args &&...args) {
    return function_ref(std::forward<Args...>(args)...);
}

DEFINE_TEST("Empty behaviour") {
    FunctionRef<void()> empty;
    CATCH_CHECK_FALSE(empty);
    CATCH_CHECK_THROWS_AS(empty(), std::bad_function_call);
    CATCH_CHECK_THROWS_AS(call_functor(empty), std::bad_function_call);

    FunctionRef<void()> copy = empty;
    CATCH_CHECK_FALSE(copy);
    CATCH_CHECK_THROWS_AS(copy(), std::bad_function_call);
    CATCH_CHECK_THROWS_AS(call_functor(copy), std::bad_function_call);

    FunctionRef<void()> copy_assign;
    copy_assign = copy;
    CATCH_CHECK_FALSE(copy_assign);
    CATCH_CHECK_THROWS_AS(copy_assign(), std::bad_function_call);
    CATCH_CHECK_THROWS_AS(call_functor(copy_assign), std::bad_function_call);
}

DEFINE_TEST("Callable is called once") {
    int counter = 0;
    auto add_to_counter = [&](int x) { return counter += x; };

    CATCH_CHECK(call_functor<int(int)>(add_to_counter, 1) == 1);
    CATCH_CHECK(counter == 1);

    FunctionRef<int(int)> functor = add_to_counter;
    CATCH_CHECK(functor(1) == 2);
    CATCH_CHECK(counter == 2);
    CATCH_CHECK(call_functor(functor, 2) == 4);
    CATCH_CHECK(counter == 4);

    FunctionRef<int(int)> copy = functor;
    CATCH_CHECK(copy(3) == 7);
    CATCH_CHECK(counter == 7);
    CATCH_CHECK(call_functor(copy, 3) == 10);
    CATCH_CHECK(counter == 10);

    FunctionRef<int(int)> copy_assign;
    copy_assign = copy;
    CATCH_CHECK(copy_assign(4) == 14);
    CATCH_CHECK(counter == 14);
    CATCH_CHECK(call_functor(copy_assign, 4) == 18);
    CATCH_CHECK(counter == 18);
}

DEFINE_TEST("Move only lambda") {
    auto counter_ptr = std::make_unique<int>();
    const int &counter = *counter_ptr;

    auto lambda = [ptr = std::move(counter_ptr)] { *ptr += 1; };
    using Lambda = std::remove_reference_t<decltype(lambda)>;
    static_assert(!std::is_copy_constructible_v<Lambda> && !std::is_copy_assignable_v<Lambda>);
    FunctionRef<void()> functor = lambda;

    CATCH_CHECK(counter == 0);
    call_functor<void()>(lambda);
    CATCH_CHECK(counter == 1);
    functor();
    CATCH_CHECK(counter == 2);
    call_functor(functor);
    CATCH_CHECK(counter == 3);
}

DEFINE_TEST("Constructible from function pointer") {
    static constexpr int (*add_one)(int) = +[](int x) { return x + 1; };

    CATCH_CHECK(call_functor<int(int)>(add_one, 1) == 2);

    FunctionRef<int(int)> functor = add_one;
    CATCH_CHECK(functor(2) == 3);
    functor = add_one;
    CATCH_CHECK(functor(3) == 4);

    // nullptr is a valid pointer
    functor = nullptr;
    CATCH_CHECK_FALSE(functor);
    functor = (int (*)(int)) nullptr;
    CATCH_CHECK_FALSE(functor);
}

DEFINE_TEST("Invocable with refs") {
    auto lambda = [](int &x) { x++; };
    FunctionRef<void(int &)> functor = lambda;

    int counter = 0;
    call_functor<void(int &)>(lambda, counter);
    CATCH_CHECK(counter == 1);
    functor(counter);
    CATCH_CHECK(counter == 2);
    call_functor(functor, counter);
    CATCH_CHECK(counter == 3);
}

DEFINE_TEST("const correctness") {
    static_assert(std::is_invocable_v<FunctionRef<void(void)>>);
    static_assert(!std::is_invocable_v<const FunctionRef<void()>>);
    static_assert(std::is_invocable_v<FunctionRef<void() const>>);
    static_assert(std::is_invocable_v<const FunctionRef<void() const>>);

    auto lambda = [](int &x) { x++; };
    FunctionRef<void(int &) const> add_one = lambda;

    int counter = 0;
    add_one(counter);
    CATCH_CHECK(counter == 1);
    std::as_const(add_one)(counter);
    CATCH_CHECK(counter == 2);
}

}  // namespace
