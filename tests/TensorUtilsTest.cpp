#include "torch/torch.h"
#include "utils/tensor_utils.h"

#include <catch2/catch.hpp>

#include <random>

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

    auto expected = torch::quantile(in.to(torch::kFloat), q, 0, false, c10::string_view("lower"));
    auto computed = dorado::utils::quantile_counting(in.to(torch::kI16), q);

    REQUIRE(torch::equal(computed, expected));
}

TEST_CASE(CUT_TAG ": test quantiles_counting", CUT_TAG) {
    auto in = torch::randint(0, 2047, 1000);
    auto q = torch::tensor({0.2, 0.9}, {torch::kFloat});

    auto expected = torch::quantile(in.to(torch::kFloat), q, 0, false, c10::string_view("lower"));
    auto computed = dorado::utils::quantile_counting(in.to(torch::kI16), q);

    REQUIRE(torch::equal(computed, expected));
}

TEST_CASE(CUT_TAG ": quantile_counting guppy comparison", CUT_TAG) {
    // Generate some (fixed) random inputs
    // These should match the equivalent test in guppy
    const float expected[] = {0, 65, -113, -11, -63};
    std::minstd_rand rng(42);
    auto between = [&rng](auto min, auto max) {
        return min + (max - min) * float(rng()) / rng.max();
    };
    // Use an input size greater than that of the datatype we're testing with (int16_t) in
    // order to flush out any bugs where it might be misused as an index
    std::vector<int16_t> input_data(314159);
    std::generate(input_data.begin(), input_data.end(),
                  [&] { return static_cast<int16_t>(between(-123, 456)); });
    std::vector<float> quantiles(std::size(expected));
    std::generate(quantiles.begin(), quantiles.end(), [&] { return between(0.f, 1.f); });

    // Run the test
    auto input_tensor = torch::tensor(at::makeArrayRef(input_data), {torch::kI16});
    auto quantiles_tensor = torch::tensor(at::makeArrayRef(quantiles), {torch::kFloat});
    auto computed = dorado::utils::quantile_counting(input_tensor, quantiles_tensor);

    // Check the output matches
    for (size_t i = 0; i < std::size(expected); i++) {
        CAPTURE(i);
        CHECK(expected[i] == computed[i].item<float>());
    }
}
