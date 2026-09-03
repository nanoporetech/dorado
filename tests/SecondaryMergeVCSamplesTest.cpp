
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <string>
#include <vector>

namespace dorado::secondary::consensus::tests {

#define TEST_GROUP "[SecondaryConsensus]"

CATCH_TEST_CASE("merge_vc_samples", TEST_GROUP) {
    struct TestCase {
        std::string test_name;
        std::vector<VariantCallingSample> in_samples;
        std::vector<VariantCallingSample> expected;
        bool expect_throw = false;
    };

    // clang-format off
    auto [test_case] = GENERATE_REF(table<TestCase>({
        TestCase{
            "Empty test",
            {}, {}, false,
        },
        TestCase{
            "Sample not valid, should throw. positions_minor not same length as positions_major.",
            {
                {
                    0,
                    {195447, 195447, 195447, 195447},
                    {},
                    at::tensor({0.0f, 1.0f, 2.0f, 3.0f}, at::kFloat),
                    at::ones({4}, at::kInt),
                },
            },
            {
            },
            true,
        },
        TestCase{
            "Merge adjacent samples split on regular major positions",
            {
                {
                    0,
                    {0, 1, 2, 3},
                    {0, 0, 0, 0},
                    at::tensor({0.0f, 1.0f, 2.0f, 3.0f}, at::kFloat),
                    at::ones({4}, at::kInt),
                },
                {
                    0,
                    {4, 5, 6, 7, 8},
                    {0, 0, 0, 0, 0},
                    at::tensor({4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, at::kFloat),
                    at::full({5}, 2, at::kInt),
                },
            },
            {
                {
                    0,
                    {0, 1, 2, 3, 4, 5, 6, 7, 8},
                    {0, 0, 0, 0, 0, 0, 0, 0, 0,},
                    at::tensor({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, at::kFloat),
                    at::tensor({1, 1, 1, 1, 2, 2, 2, 2, 2}, at::kInt),
                },
            },
            false,
        },
        TestCase{
            "Merge adjacent samples which are split on minor coordinates",
            {
                {
                    0,
                    {195447, 195447, 195447, 195447},
                    {0, 1, 2, 3},
                    at::tensor({0.0f, 1.0f, 2.0f, 3.0f}, at::kFloat),
                    at::ones({4}, at::kInt),
                },
                {
                    0,
                    {195447, 195448, 195449, 195450, 195451},
                    {4, 0, 0, 0, 0},
                    at::tensor({4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, at::kFloat),
                    at::full({5}, 2, at::kInt),
                },
            },
            {
                {
                    0,
                    {195447, 195447, 195447, 195447, 195447, 195448, 195449, 195450, 195451},
                    {0, 1, 2, 3, 4, 0, 0, 0, 0},
                    at::tensor({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, at::kFloat),
                    at::tensor({1, 1, 1, 1, 2, 2, 2, 2, 2}, at::kInt),
                },
            },
            false,
        },
        TestCase{
            "Do not merge samples which are 1bp or more apart.",
            {
                {
                    0,
                    {0, 1, 2, 3},
                    {0, 0, 0, 0},
                    at::tensor({0.0f, 1.0f, 2.0f, 3.0f}, at::kFloat),
                    at::ones({4}, at::kInt),
                },
                {
                    0,
                    {5, 6, 7, 8},
                    {0, 0, 0, 0},
                    at::tensor({5.0f, 6.0f, 7.0f, 8.0f}, at::kFloat),
                    at::full({4}, 2, at::kInt),
                },
            },
            {
                {
                    0,
                    {0, 1, 2, 3},
                    {0, 0, 0, 0},
                    at::tensor({0.0f, 1.0f, 2.0f, 3.0f}, at::kFloat),
                    at::ones({4}, at::kInt),
                },
                {
                    0,
                    {5, 6, 7, 8},
                    {0, 0, 0, 0},
                    at::tensor({5.0f, 6.0f, 7.0f, 8.0f}, at::kFloat),
                    at::full({4}, 2, at::kInt),
                },
            },
            false,
        },
        TestCase{
            "Do not merge adjacent major positions when the next sample starts at a minor position",
            {
                {
                    5,
                    {11936695},
                    {14},
                    at::tensor({0.0f}, at::kFloat),
                    at::ones({1}, at::kInt),
                },
                {
                    5,
                    {11936696, 11936697},
                    {62, 0},
                    at::tensor({1.0f, 2.0f}, at::kFloat),
                    at::full({2}, 2, at::kInt),
                },
            },
            {
                {
                    5,
                    {11936695},
                    {14},
                    at::tensor({0.0f}, at::kFloat),
                    at::ones({1}, at::kInt),
                },
                {
                    5,
                    {11936696, 11936697},
                    {62, 0},
                    at::tensor({1.0f, 2.0f}, at::kFloat),
                    at::full({2}, 2, at::kInt),
                },
            },
            false,
        },
        TestCase{
            "Do not merge samples which have same major position but minor positions are not adjacent",
            {
                {
                    0,
                    {0, 1, 2, 3, 4, 4},
                    {0, 0, 0, 0, 0, 1},
                    at::tensor({0.0f, 1.0f, 2.0f, 3.0f, 3.1, 3.2}, at::kFloat),
                    at::ones({6}, at::kInt),
                },
                {
                    0,
                    {4, 5, 6, 7, 8},
                    {5, 0, 0, 0, 0},
                    at::tensor({4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, at::kFloat),
                    at::full({5}, 2, at::kInt),
                },
            },
            {
                {
                    0,
                    {0, 1, 2, 3, 4, 4},
                    {0, 0, 0, 0, 0, 1},
                    at::tensor({0.0f, 1.0f, 2.0f, 3.0f, 3.1, 3.2}, at::kFloat),
                    at::ones({6}, at::kInt),
                },
                {
                    0,
                    {4, 5, 6, 7, 8},
                    {5, 0, 0, 0, 0},
                    at::tensor({4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, at::kFloat),
                    at::full({5}, 2, at::kInt),
                },
            },
            false,
        },
    }));
    // clang-format on

    CATCH_INFO(TEST_GROUP << " Test name: " << test_case.test_name);

    if (test_case.expect_throw) {
        CATCH_CHECK_THROWS(merge_vc_samples(test_case.in_samples));
    } else {
        const std::vector<VariantCallingSample> result = merge_vc_samples(test_case.in_samples);
        CATCH_CHECK(test_case.expected == result);
    }
}

CATCH_TEST_CASE("variant_calling_sample_less orders by full start coordinate", TEST_GROUP) {
    // clang-format off
    const VariantCallingSample major_start_only{
            0,
            {195447, 195447, 195448},
            {1, 2, 0},
            at::ones({3, 1}, at::kFloat),
            at::ones({3}, at::kInt),
    };
    const VariantCallingSample true_predecessor{
            0,
            {195447, 195448},
            {0, 0},
            at::ones({2, 1}, at::kFloat),
            at::full({2}, 2, at::kInt),
    };
    // clang-format on

    CATCH_CHECK(variant_calling_sample_less(true_predecessor, major_start_only));
    CATCH_CHECK_FALSE(variant_calling_sample_less(major_start_only, true_predecessor));
}

CATCH_TEST_CASE("variant_calling_sample_less prefers longer sample for identical starts",
                TEST_GROUP) {
    // clang-format off
    const VariantCallingSample longer{
            0,
            {195447, 195447, 195448},
            {0, 1, 0},
            at::ones({3, 1}, at::kFloat),
            at::ones({3}, at::kInt),
    };
    const VariantCallingSample shorter{
            0,
            {195447, 195448},
            {0, 0},
            at::ones({2, 1}, at::kFloat),
            at::full({2}, 2, at::kInt),
    };
    // clang-format on

    CATCH_CHECK(variant_calling_sample_less(longer, shorter));
    CATCH_CHECK_FALSE(variant_calling_sample_less(shorter, longer));
}

}  // namespace dorado::secondary::consensus::tests
