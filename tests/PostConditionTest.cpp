#include "utils/PostCondition.h"

#include <catch2/catch_test_macros.hpp>

#define CUT_TAG "[PostCondition]"
#define DEFINE_TEST(name) TEST_CASE(CUT_TAG " " name, CUT_TAG)

using dorado::utils::PostCondition;

DEFINE_TEST("Condition isn't triggered right away") {
    int counter = 0;
    auto on_scope_end = PostCondition([&] { counter++; });
    CHECK(counter == 0);
}

DEFINE_TEST("Condition is triggered on scope end") {
    int counter = 0;
    {
        auto on_scope_end = PostCondition([&] { counter++; });
        CHECK(counter == 0);
    }
    CHECK(counter == 1);
}

DEFINE_TEST("Multiple scopes") {
    int counter = 0;
    {
        const int outer_increment = 1;
        auto outer_scope = PostCondition([&] { counter -= outer_increment; });
        CHECK(counter == 0);
        counter += outer_increment;
        CHECK(counter == outer_increment);
        {
            const int inner_increment = 2;
            auto inner_scope = PostCondition([&] { counter -= inner_increment; });
            CHECK(counter == outer_increment);
            counter += inner_increment;
            CHECK(counter == outer_increment + inner_increment);
        }
        CHECK(counter == outer_increment);
    }
    CHECK(counter == 0);
}
