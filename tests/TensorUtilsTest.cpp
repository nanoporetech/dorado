#include "torch_utils/tensor_utils.h"

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <spdlog/spdlog.h>
#include <torch/torch.h>

#include <cstdlib>
#include <random>

#define CUT_TAG "[TensorUtils]"

CATCH_TEST_CASE(CUT_TAG ": test quartiles", CUT_TAG) {
    auto in = torch::rand(1000, {torch::kFloat});
    auto q = torch::tensor({0.25, 0.5, 0.75}, {torch::kFloat});

    auto expected = torch::quantile(in, q, 0, false, c10::string_view("lower"));
    auto computed = dorado::utils::quantile(in, q);

    CATCH_REQUIRE(torch::equal(computed, expected));
}

CATCH_TEST_CASE(CUT_TAG ": test quartiles reversed", CUT_TAG) {
    auto in = torch::rand(1000, {torch::kFloat});
    auto q = torch::tensor({0.75, 0.5, 0.25}, {torch::kFloat});

    auto expected = torch::quantile(in, q, 0, false, c10::string_view("lower"));
    auto computed = dorado::utils::quantile(in, q);

    CATCH_REQUIRE(torch::equal(computed, expected));
}

CATCH_TEST_CASE(CUT_TAG ": test quantiles", CUT_TAG) {
    auto in = torch::rand(1000, {torch::kFloat});
    auto q = torch::tensor({0.2, 0.9}, {torch::kFloat});

    auto expected = torch::quantile(in, q, 0, false, c10::string_view("lower"));
    auto computed = dorado::utils::quantile(in, q);

    CATCH_REQUIRE(torch::equal(computed, expected));
}

CATCH_TEST_CASE(CUT_TAG ": test quartiles_counting", CUT_TAG) {
    auto in = torch::randint(0, 2047, 1000);
    auto q = torch::tensor({0.25, 0.5, 0.75}, {torch::kFloat});

    auto expected = torch::quantile(in.to(torch::kFloat), q, 0, false, c10::string_view("lower"));
    auto computed = dorado::utils::quantile_counting(in.to(torch::kI16), q);

    CATCH_REQUIRE(torch::equal(computed, expected));
}

CATCH_TEST_CASE(CUT_TAG ": test quantiles_counting", CUT_TAG) {
    auto in = torch::randint(0, 2047, 1000);
    auto q = torch::tensor({0.2, 0.9}, {torch::kFloat});

    auto expected = torch::quantile(in.to(torch::kFloat), q, 0, false, c10::string_view("lower"));
    auto computed = dorado::utils::quantile_counting(in.to(torch::kI16), q);

    CATCH_REQUIRE(torch::equal(computed, expected));
}

#if DORADO_ENABLE_BENCHMARK_TESTS
CATCH_TEST_CASE(CUT_TAG ": quantile benchmark", CUT_TAG) {
    const auto size = GENERATE(1000, 5000, 10000, 100000);
    CATCH_CAPTURE(size);

    const auto x_int = at::randint(0, 2047, size);
    const auto x_float = x_int.to(at::ScalarType::Float);
    const auto x_short = x_int.to(at::ScalarType::Short);
    const auto q = at::tensor({0.2, 0.9}, {at::ScalarType::Float});

    CATCH_BENCHMARK(fmt::format("torch quantile (float/{})", size)) {
        return at::quantile(x_float, q);
    };
    CATCH_BENCHMARK(fmt::format("our quantile (float/{})", size)) {
        return dorado::utils::quantile(x_float, q);
    };
    CATCH_BENCHMARK(fmt::format("our quantile (short/{})", size)) {
        return dorado::utils::quantile_counting(x_short, q);
    };
}
#endif  // DORADO_ENABLE_BENCHMARK_TESTS

CATCH_TEST_CASE(CUT_TAG ": convert_f32_to_f16", CUT_TAG) {
    torch::manual_seed(42);
    srand(42);

    for (int i = 0; i < 10; ++i) {
        const int num_elems = rand() % 100;
        const auto elems_f32 = torch::rand({num_elems}, torch::kFloat32);
        const auto elems_torch_f16 = elems_f32.to(torch::kHalf);
        auto elems_converted_f16 = torch::zeros({num_elems}, torch::kHalf);
        dorado::utils::convert_f32_to_f16(elems_converted_f16.data_ptr<c10::Half>(),
                                          elems_f32.data_ptr<float>(), num_elems);
        const float kRelTolerance = 0.0f;
        const float kAbsTolerance = 0.0f;
        CATCH_CHECK(torch::allclose(elems_torch_f16, elems_converted_f16, kRelTolerance,
                                    kAbsTolerance));
    }

#if DORADO_ENABLE_BENCHMARK_TESTS
    {
        const auto num_elems = GENERATE(1'000, 100'000, 10'000'000);
        const auto elems_f32 = torch::rand({num_elems}, torch::kFloat32);
        auto elems_converted_f16 = torch::zeros({num_elems}, torch::kHalf);

        CATCH_BENCHMARK("torch convert " + std::to_string(num_elems)) {
            elems_f32.to(torch::kHalf);
        };

        CATCH_BENCHMARK("our convert " + std::to_string(num_elems)) {
            dorado::utils::convert_f32_to_f16(elems_converted_f16.data_ptr<c10::Half>(),
                                              elems_f32.data_ptr<float>(), num_elems);
        };
    }
#endif  // DORADO_ENABLE_BENCHMARK_TESTS
}

CATCH_TEST_CASE(CUT_TAG ": copy_tensor_elems", CUT_TAG) {
    torch::manual_seed(42);
    srand(42);

    for (auto src_dtype : {torch::kFloat16, torch::kFloat32}) {
        for (auto dest_dtype : {torch::kFloat16, torch::kFloat32}) {
            for (int i = 0; i < 10; ++i) {
                const int src_size = rand() % 1000;
                const at::Tensor src_tensor = torch::rand({src_size}, src_dtype);
                const int dest_size = src_size + rand() % 1000;
                const at::Tensor orig_dest_tensor = torch::rand({dest_size}, dest_dtype);

                const int dest_offset = rand() % dest_size;
                const int src_offset = rand() % src_size;
                const int max_count = std::min(dest_size - dest_offset, src_size - src_offset);
                const int count = rand() % max_count;

                using torch::indexing::Slice;
                auto torch_result = orig_dest_tensor.clone();
                torch_result.index_put_({Slice(dest_offset, dest_offset + count)},
                                        src_tensor.index({Slice(src_offset, src_offset + count)}));

                auto copy_elems_result = orig_dest_tensor.clone();
                dorado::utils::copy_tensor_elems(copy_elems_result, dest_offset, src_tensor,
                                                 src_offset, count);
                const float kRelTolerance = 0.0f;
                const float kAbsTolerance = 0.0f;
                CATCH_CHECK(torch::allclose(torch_result, copy_elems_result, kRelTolerance,
                                            kAbsTolerance));
            }
        }
    }
}
