#include "utils/PostCondition.h"

#include <catch2/catch_test_macros.hpp>

#define CUT_TAG "[PostCondition]"
#define DEFINE_TEST(name) CATCH_TEST_CASE(CUT_TAG " " name, CUT_TAG)

using dorado::utils::PostCondition;

DEFINE_TEST("Condition isn't triggered right away") {
    int counter = 0;
    auto on_scope_end = PostCondition([&] { counter++; });
    CATCH_CHECK(counter == 0);
}

DEFINE_TEST("Condition is triggered on scope end") {
    int counter = 0;
    {
        auto on_scope_end = PostCondition([&] { counter++; });
        CATCH_CHECK(counter == 0);
    }
    CATCH_CHECK(counter == 1);
}

DEFINE_TEST("Multiple scopes") {
    int counter = 0;
    {
        const int outer_increment = 1;
        auto outer_scope = PostCondition([&] { counter -= outer_increment; });
        CATCH_CHECK(counter == 0);
        counter += outer_increment;
        CATCH_CHECK(counter == outer_increment);
        {
            const int inner_increment = 2;
            auto inner_scope = PostCondition([&] { counter -= inner_increment; });
            CATCH_CHECK(counter == outer_increment);
            counter += inner_increment;
            CATCH_CHECK(counter == outer_increment + inner_increment);
        }
        CATCH_CHECK(counter == outer_increment);
    }
    CATCH_CHECK(counter == 0);
}
