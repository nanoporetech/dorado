#include "torch/torch.h"
#include "utils/tensor_utils.h"

#include <catch2/catch.hpp>

#define CUT_TAG "[TensorUtils]"

TEST_CASE(CUT_TAG ": test quartiles", CUT_TAG) {
    auto in = torch::rand(1000, {torch::kFloat});
    auto q = torch::tensor({0.25, 0.5, 0.75}, {torch::kFloat});

    auto expected = torch::quantile(in, q, 0, false, c10::string_view("lower"));
    auto computed = dorado::utils::quantile(in, q);

    REQUIRE(torch::equal(computed, expected));
}

TEST_CASE(CUT_TAG ": test quartiles reversed", CUT_TAG) {
    auto in = torch::rand(1000, {torch::kFloat});
    auto q = torch::tensor({0.75, 0.5, 0.25}, {torch::kFloat});

    auto expected = torch::quantile(in, q, 0, false, c10::string_view("lower"));
    auto computed = dorado::utils::quantile(in, q);

    REQUIRE(torch::equal(computed, expected));
}

TEST_CASE(CUT_TAG ": test quantiles", CUT_TAG) {
    auto in = torch::rand(1000, {torch::kFloat});
    auto q = torch::tensor({0.2, 0.9}, {torch::kFloat});

    auto expected = torch::quantile(in, q, 0, false, c10::string_view("lower"));
    auto computed = dorado::utils::quantile(in, q);

    REQUIRE(torch::equal(computed, expected));
}

TEST_CASE(CUT_TAG ": test quartiles_counting", CUT_TAG) {
    auto in = torch::randint(0, 2047, 1000);
    auto q = torch::tensor({0.25, 0.5, 0.75}, {torch::kFloat});

    auto expected = torch::quantile(in, q, 0, false, c10::string_view("lower"));
    auto computed = dorado::utils::quantile_counting(in.to(torch::kI16), q);

    REQUIRE(torch::equal(computed, expected));
}

TEST_CASE(CUT_TAG ": test quantiles_counting", CUT_TAG) {
    auto in = torch::randint(0, 2047, 1000);
    auto q = torch::tensor({0.2, 0.9}, {torch::kFloat});

    auto expected = torch::quantile(in, q, 0, false, c10::string_view("lower"));
    auto computed = dorado::utils::quantile_counting(in.to(torch::kI16), q);

    REQUIRE(torch::equal(computed, expected));
}
