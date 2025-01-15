#include "utils/math_utils.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#define CUT_TAG "[MathUtils]"

TEST_CASE(CUT_TAG ": test quantiles", CUT_TAG) {
    std::vector<double> in = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    auto quartiles = dorado::utils::quantiles(in, {0.25, 0.5, 0.75});

    std::vector<double> expected_values{3.5, 6.0, 8.5};
    REQUIRE(quartiles.size() == expected_values.size());
    for (size_t i = 0; i < quartiles.size(); ++i) {
        REQUIRE(quartiles[i] == Catch::Approx(expected_values[i]));
    }
}

TEST_CASE(CUT_TAG ": test linear_regression", CUT_TAG) {
    std::vector<double> x = {1, 2, 4, 5, 10, 20};
    std::vector<double> y = {4, 6, 12, 15, 34, 68};

    auto [m, b, r] = dorado::utils::linear_regression(x, y);

    double expected_m = 3.43651, expected_b = -0.888889, expected_r = 0.999192;
    REQUIRE(m == Catch::Approx(expected_m));
    REQUIRE(b == Catch::Approx(expected_b));
    REQUIRE(r == Catch::Approx(expected_r));
}

TEST_CASE(CUT_TAG ": test equality within tolerance", CUT_TAG) {
    SECTION("Check for ints") {
        CHECK(dorado::utils::eq_with_tolerance(100, 110, 20) == true);
        CHECK(dorado::utils::eq_with_tolerance(110, 100, 5) == false);
    }
    SECTION("Check for floats") {
        CHECK(dorado::utils::eq_with_tolerance(100.f, 101.f, 1.1f) == true);
        CHECK(dorado::utils::eq_with_tolerance(100.f, 101.f, 0.9f) == false);
    }
}
