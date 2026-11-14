#include "secondary/consensus/sample.h"
#include "secondary/consensus/variant_calling_sample.h"
#include "secondary/features/decoder_base.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace dorado::secondary::consensus::tests {

#define TEST_GROUP "[SecondaryConsensus]"

CATCH_TEST_CASE("VariantCallingSample::validate checks basecalls and read IDs", TEST_GROUP) {
    const auto make_sample = [](const at::Tensor& basecalls, std::vector<std::string> read_ids_left,
                                std::vector<std::string> read_ids_right) {
        return VariantCallingSample{
                .seq_id = 0,
                .positions_major = {0, 1, 2},
                .positions_minor = {0, 0, 0},
                .logits = at::ones({3, 1}, at::kFloat),
                .depth = at::ones({3}, at::kInt),
                .read_ids_left = std::move(read_ids_left),
                .read_ids_right = std::move(read_ids_right),
                .basecalls = basecalls,
        };
    };

    CATCH_SECTION("valid rank-3 basecalls and matching read IDs") {
        auto sample = make_sample(at::ones({3, 2}, at::kChar), {"left_0", "left_1"},
                                  {"right_0", "right_1"});
        CATCH_REQUIRE_NOTHROW(sample.validate());
    }

    CATCH_SECTION("basecall length mismatch") {
        auto sample_too_short = make_sample(at::ones({1, 2}, at::kChar), {"left_0", "left_1"},
                                            {"right_0", "right_1"});
        CATCH_REQUIRE_THROWS(sample_too_short.validate());
        auto sample_too_long = make_sample(at::ones({4, 2}, at::kChar), {"left_0", "left_1"},
                                           {"right_0", "right_1"});
        CATCH_REQUIRE_THROWS(sample_too_long.validate());
    }

    CATCH_SECTION("empty left read IDs") {
        auto sample = make_sample(at::ones({3, 2}, at::kChar), {}, {"right_0", "right_1"});
        CATCH_REQUIRE_THROWS(sample.validate());
    }

    CATCH_SECTION("empty right read IDs") {
        auto sample = make_sample(at::ones({3, 2}, at::kChar), {"left_0", "left_1"}, {});
        CATCH_REQUIRE_THROWS(sample.validate());
    }

    CATCH_SECTION("undersized and oversized left read IDs") {
        auto undersized =
                make_sample(at::ones({3, 2}, at::kChar), {"left_0"}, {"right_0", "right_1"});
        CATCH_REQUIRE_THROWS(undersized.validate());

        auto oversized = make_sample(at::ones({3, 2}, at::kChar), {"left_0", "left_1", "left_2"},
                                     {"right_0", "right_1"});
        CATCH_REQUIRE_THROWS(oversized.validate());
    }

    CATCH_SECTION("undersized and oversized right read IDs") {
        auto undersized =
                make_sample(at::ones({3, 2}, at::kChar), {"left_0", "left_1"}, {"right_0"});
        CATCH_REQUIRE_THROWS(undersized.validate());

        auto oversized = make_sample(at::ones({3, 2}, at::kChar), {"left_0", "left_1"},
                                     {"right_0", "right_1", "right_2"});
        CATCH_REQUIRE_THROWS(oversized.validate());
    }

    CATCH_SECTION("basecalls wrong dimensions") {
        auto sample1d =
                make_sample(at::ones({3}, at::kChar), {"left_0", "left_1"}, {"right_0", "right_1"});
        CATCH_REQUIRE_THROWS(sample1d.validate());

        auto sample3d = make_sample(at::ones({3, 2, 1}, at::kChar), {"left_0", "left_1"},
                                    {"right_0", "right_1"});
        CATCH_REQUIRE_THROWS(sample3d.validate());
    }

    CATCH_SECTION("undefined basecalls and empty read IDs") {
        auto sample = make_sample(at::Tensor{}, {}, {});
        CATCH_REQUIRE_NOTHROW(sample.validate());
    }
}

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
                    {},
                    {}
                },
            },
            {
            },
            true,
        },
        TestCase{
            "Valid single sample with basecalls is returned unchanged",
            {
                {
                    0,
                    {0, 1},
                    {0, 0},
                    at::tensor({0.0f, 1.0f}, at::kFloat),
                    at::ones({2}, at::kInt),
                    {"left_0", "left_1"},
                    {"right_0", "right_1"},
                    at::arange(4, at::kChar).reshape({2, 2}),
                },
            },
            {
                {
                    0,
                    {0, 1},
                    {0, 0},
                    at::tensor({0.0f, 1.0f}, at::kFloat),
                    at::ones({2}, at::kInt),
                    {"left_0", "left_1"},
                    {"right_0", "right_1"},
                    at::arange(4, at::kChar).reshape({2, 2}),
                },
            },
            false,
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
                    {"left_read_1", "left_read_2"},
                    {"right_read_1", "right_read_2"},
                },
                {
                    0,
                    {4, 5, 6, 7, 8},
                    {0, 0, 0, 0, 0},
                    at::tensor({4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, at::kFloat),
                    at::full({5}, 2, at::kInt),
                    {"left_read_3", "left_read_4"},
                    {"right_read_3", "right_read_4"},
                },
            },
            {
                {
                    0,
                    {0, 1, 2, 3, 4, 5, 6, 7, 8},
                    {0, 0, 0, 0, 0, 0, 0, 0, 0,},
                    at::tensor({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, at::kFloat),
                    at::tensor({1, 1, 1, 1, 2, 2, 2, 2, 2}, at::kInt),
                    {"left_read_1", "left_read_2"},
                    {"right_read_3", "right_read_4"},
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
                    {"left_read_1", "left_read_2"},
                    {"mid_read_1", "mid_read_2"},
                },
                {
                    0,
                    {195447, 195448, 195449, 195450, 195451},
                    {4, 0, 0, 0, 0},
                    at::tensor({4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, at::kFloat),
                    at::full({5}, 2, at::kInt),
                    {"mid_read_1", "mid_read_2"},
                    {"right_read_3", "right_read_4"},
                },
            },
            {
                {
                    0,
                    {195447, 195447, 195447, 195447, 195447, 195448, 195449, 195450, 195451},
                    {0, 1, 2, 3, 4, 0, 0, 0, 0},
                    at::tensor({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, at::kFloat),
                    at::tensor({1, 1, 1, 1, 2, 2, 2, 2, 2}, at::kInt),
                    {"left_read_1", "left_read_2"},
                    {"right_read_3", "right_read_4"},
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
                    {"read_id1"},
                    {"read_id2"}
                },
                {
                    0,
                    {5, 6, 7, 8},
                    {0, 0, 0, 0},
                    at::tensor({5.0f, 6.0f, 7.0f, 8.0f}, at::kFloat),
                    at::full({4}, 2, at::kInt),
                    {"read_id3"},
                    {"read_id4"}
                },
            },
            {
                {
                    0,
                    {0, 1, 2, 3},
                    {0, 0, 0, 0},
                    at::tensor({0.0f, 1.0f, 2.0f, 3.0f}, at::kFloat),
                    at::ones({4}, at::kInt),
                    {"read_id1"},
                    {"read_id2"}
                },
                {
                    0,
                    {5, 6, 7, 8},
                    {0, 0, 0, 0},
                    at::tensor({5.0f, 6.0f, 7.0f, 8.0f}, at::kFloat),
                    at::full({4}, 2, at::kInt),
                    {"read_id3"},
                    {"read_id4"}
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
                    {},
                    {},
                },
                {
                    0,
                    {4, 5, 6, 7, 8},
                    {5, 0, 0, 0, 0},
                    at::tensor({4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, at::kFloat),
                    at::full({5}, 2, at::kInt),
                    {},
                    {},
                },
            },
            {
                {
                    0,
                    {0, 1, 2, 3, 4, 4},
                    {0, 0, 0, 0, 0, 1},
                    at::tensor({0.0f, 1.0f, 2.0f, 3.0f, 3.1, 3.2}, at::kFloat),
                    at::ones({6}, at::kInt),
                    {},
                    {},
                },
                {
                    0,
                    {4, 5, 6, 7, 8},
                    {5, 0, 0, 0, 0},
                    at::tensor({4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, at::kFloat),
                    at::full({5}, 2, at::kInt),
                    {},
                    {},
                },
            },
            false,
        },
        TestCase{
            "Merge adjacent basecalls with matching read IDs and no padding",
            {
                {
                    0,
                    {0, 1},
                    {0, 0},
                    at::tensor({0.0f, 1.0f}, at::kFloat),
                    at::ones({2}, at::kInt),
                    {"left_0", "left_1"},
                    {"middle_0", "middle_1"},
                    at::arange(4, at::kChar).reshape({2, 2}),
                },
                {
                    0,
                    {2, 3},
                    {0, 0},
                    at::tensor({2.0f, 3.0f}, at::kFloat),
                    at::full({2}, 2, at::kInt),
                    {"middle_0", "middle_1"},
                    {"right_0", "right_1"},
                    at::arange(4, 8, at::kChar).reshape({2, 2}),
                },
            },
            {
                {
                    0,
                    {0, 1, 2, 3},
                    {0, 0, 0, 0},
                    at::tensor({0.0f, 1.0f, 2.0f, 3.0f}, at::kFloat),
                    at::tensor({1, 1, 2, 2}, at::kInt),
                    {"left_0", "left_1"},
                    {"right_0", "right_1"},
                    at::arange(8, at::kChar).reshape({4, 2}),
                },
            },
            false,
        },
        TestCase{
            "Merge adjacent basecalls with reorders based on read IDs, no padding",
            {
                {
                    0,
                    {0, 1},
                    {0, 0},
                    at::tensor({0.0f, 1.0f}, at::kFloat),
                    at::ones({2}, at::kInt),
                    {"left_0", "left_1"},
                    {"middle_0", "middle_1"},
                    at::arange(4, at::kChar).reshape({2, 2}),
                },
                {
                    0,
                    {2, 3},
                    {0, 0},
                    at::tensor({2.0f, 3.0f}, at::kFloat),
                    at::full({2}, 2, at::kInt),
                    {"middle_1", "middle_0"},
                    {"right_0", "right_1"},
                    at::arange(4, 8, at::kChar).reshape({2, 2}),
                },
            },
            {
                {
                    0,
                    {0, 1, 2, 3},
                    {0, 0, 0, 0},
                    at::tensor({0.0f, 1.0f, 2.0f, 3.0f}, at::kFloat),
                    at::tensor({1, 1, 2, 2}, at::kInt),
                    {"left_0", "left_1"},
                    {"right_1", "right_0"},
                    at::tensor({0, 1, 2, 3, 5, 4, 7, 6}, at::kChar).reshape({4, 2}),
                },
            },
            false,
        },
        TestCase{
            "Merge adjacent samples re-orders basecalls arrays and pads, right depth > left depth",
            {
                {
                    0,
                    {0, 1, 2, 3},
                    {0, 0, 0, 0},
                    at::tensor({0.0f, 1.0f, 2.0f, 3.0f}, at::kFloat),
                    at::ones({4}, at::kInt),
                    {"read_id1", "read_id2"},
                    {"read_id3", "read_id4"},
                    at::arange(8, at::kChar).reshape({4, 2}),
                },
                {
                    0,
                    {4, 5, 6, 7, 8},
                    {0, 0, 0, 0, 0},
                    at::tensor({4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, at::kFloat),
                    at::full({5}, 2, at::kInt),
                    {"read_id3", "read_id4", "read_id5"},
                    {"read_id6", "read_id7", "read_id5"},
                    at::arange(15, at::kChar).reshape({5, 3}),
                },
            },
            {
                {
                    0,
                    {0, 1, 2, 3, 4, 5, 6, 7, 8},
                    {0, 0, 0, 0, 0, 0, 0, 0, 0,},
                    at::tensor({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, at::kFloat),
                    at::tensor({1, 1, 1, 1, 2, 2, 2, 2, 2}, at::kInt),
                    {"read_id1", "read_id2", "__blank"},
                    {"read_id6", "read_id7", "read_id5"},
                    at::tensor({ 0,  1,  0,
                                 2,  3,  0,
                                 4,  5,  0,
                                 6,  7,  0,
                                 0,  1,  2,
                                 3,  4,  5,
                                 6,  7,  8,
                                 9, 10, 11,
                                12, 13, 14}, at::kChar).reshape({9, 3}),
                },
            },
            false,
        },
        TestCase{
            "Merge adjacent samples re-orders basecalls arrays and pads, right depth < left depth",
            {
                {
                    0,
                    {0, 1, 2, 3},
                    {0, 0, 0, 0},
                    at::tensor({0.0f, 1.0f, 2.0f, 3.0f}, at::kFloat),
                    at::full({4}, 3, at::kInt),
                    {"read_id1", "read_id2", "read_id3"},
                    {"read_id1", "read_id2", "read_id3"},
                    at::arange(12, at::kChar).reshape({4, 3}),
                },
                {
                    0,
                    {4, 5, 6, 7, 8},
                    {0, 0, 0, 0, 0},
                    at::tensor({4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, at::kFloat),
                    at::full({5}, 2, at::kInt),
                    {"read_id2", "read_id3"},
                    {"read_id4", "read_id5"},
                    at::arange(10, at::kChar).reshape({5, 2}),
                },
            },
            {
                {
                    0,
                    {0, 1, 2, 3, 4, 5, 6, 7, 8},
                    {0, 0, 0, 0, 0, 0, 0, 0, 0},
                    at::tensor({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, at::kFloat),
                    at::tensor({3, 3, 3, 3, 2, 2, 2, 2, 2}, at::kInt),
                    {"read_id1", "read_id2", "read_id3"},
                    {"__blank", "read_id4", "read_id5"},
                    at::tensor({ 0,  1,  2,
                                 3,  4,  5,
                                 6,  7,  8,
                                 9, 10, 11,
                                 0,  0,  1,
                                 0,  2,  3,
                                 0,  4,  5,
                                 0,  6,  7,
                                 0,  8,  9}, at::kChar).reshape({9, 3}),
                },
            },
            false,
        },
        TestCase{
            "Merge adjacent samples re-orders basecalls arrays based on read ids",
            {
                {
                    0,
                    {0, 1, 2, 3},
                    {0, 0, 0, 0},
                    at::tensor({0.0f, 1.0f, 2.0f, 3.0f}, at::kFloat),
                    at::ones({4}, at::kInt),
                    {"read_id1", "read_id2"},
                    {"read_id3", "read_id4"},
                    at::arange(8, at::kChar).reshape({4, 2}),
                },
                {
                    0,
                    {4, 5, 6, 7, 8},
                    {0, 0, 0, 0, 0},
                    at::tensor({4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, at::kFloat),
                    at::full({5}, 2, at::kInt),
                    {"read_id4", "read_id5", "read_id3"},
                    {"read_id6", "read_id5", "read_id7"},
                    at::arange(15, at::kChar).reshape({5, 3}),
                },
            },
            {
                {
                    0,
                    {0, 1, 2, 3, 4, 5, 6, 7, 8},
                    {0, 0, 0, 0, 0, 0, 0, 0, 0},
                    at::tensor({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, at::kFloat),
                    at::tensor({1, 1, 1, 1, 2, 2, 2, 2, 2}, at::kInt),
                    {"read_id1", "read_id2", "__blank"},
                    {"read_id7", "read_id6", "read_id5"},
                    at::tensor({ 0,  1,  0,
                                 2,  3,  0,
                                 4,  5,  0,
                                 6,  7,  0,
                                 2,  0,  1,
                                 5,  3,  4,
                                 8,  6,  7,
                                11,  9, 10,
                                14, 12, 13}, at::kChar).reshape({9, 3}),
                },
            },
            false,
        },
        TestCase{
            "Invalid combination of basecall arrays, raises",
            {
                {
                    0,
                    {0, 1, 2, 3},
                    {0, 0, 0, 0},
                    at::tensor({0.0f, 1.0f, 2.0f, 3.0f}, at::kFloat),
                    at::ones({4}, at::kInt),
                    {"read1", "read2", "read3", "read4"},
                    {"read1", "read2", "read3", "read4"},
                    at::ones({4, 4}, at::kChar),
                },
                {
                    0,
                    {4, 5, 6, 7, 8},
                    {0, 0, 0, 0, 0},
                    at::tensor({4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, at::kFloat),
                    at::full({5}, 2, at::kInt),
                    {},
                    {},
                    at::empty({}, at::kChar),
                },
            },
            {
            },
            true,
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

CATCH_TEST_CASE("slice_vc_sample preserves boundary IDs and basecalls", TEST_GROUP) {
    const VariantCallingSample sample{
            .seq_id = 0,
            .positions_major = {0, 1, 2, 3},
            .positions_minor = {0, 0, 0, 0},
            .logits = at::arange(8, at::kFloat).reshape({4, 2}),
            .depth = at::tensor({1, 2, 3, 4}, at::kInt),
            .read_ids_left = {"left_0", "left_1"},
            .read_ids_right = {"right_0", "right_1"},
            .basecalls = at::arange(8, at::kChar).reshape({4, 2}),
    };
    const std::vector<std::string> placeholder_ids = placeholder_read_ids(2);

    CATCH_SECTION("internal slice") {
        const VariantCallingSample result = slice_vc_sample(sample, 1, 3, true);

        CATCH_CHECK(result.read_ids_left == placeholder_ids);
        CATCH_CHECK(result.read_ids_right == placeholder_ids);
        CATCH_CHECK(result.basecalls.equal(sample.basecalls.slice(0, 1, 3, true)));
    }

    CATCH_SECTION("left boundary slice") {
        const VariantCallingSample result = slice_vc_sample(sample, 0, 2, true);

        CATCH_CHECK(result.read_ids_left == sample.read_ids_left);
        CATCH_CHECK(result.read_ids_right == placeholder_ids);
        CATCH_CHECK(result.basecalls.equal(sample.basecalls.slice(0, 0, 2)));
    }

    CATCH_SECTION("right boundary slice") {
        const VariantCallingSample result = slice_vc_sample(sample, 2, 4, true);

        CATCH_CHECK(result.read_ids_left == placeholder_ids);
        CATCH_CHECK(result.read_ids_right == sample.read_ids_right);
        CATCH_CHECK(result.basecalls.equal(sample.basecalls.slice(0, 2, 4)));
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

CATCH_TEST_CASE("trim_vc_samples reuses tensors for an unchanged sample", TEST_GROUP) {
    const VariantCallingSample sample{
            0,
            {0, 1, 2, 3},
            {0, 0, 0, 0},
            at::arange(8, at::kFloat).reshape({4, 2}),
            at::tensor({1, 2, 3, 4}, at::kInt),
    };

    const std::vector<VariantCallingSample> result = trim_vc_samples({sample}, {0});

    CATCH_REQUIRE(result.size() == 1);
    CATCH_CHECK(result.front().logits.data_ptr() == sample.logits.data_ptr());
    CATCH_CHECK(result.front().depth.data_ptr() == sample.depth.data_ptr());
}

CATCH_TEST_CASE("split_vc_samples_on_boundaries", TEST_GROUP) {
    const DecoderBase decoder(LabelSchemeType::HAPLOID);
    const std::string symbols = decoder.get_label_scheme_symbols();
    const std::string draft = "ACGTACGTAC";

    // One-hot logits for the given consensus. Sample covers the whole draft.
    const auto make_sample = [&](const std::string_view cons) {
        const int64_t n = std::ssize(cons);
        at::Tensor logits = at::zeros({n, static_cast<int64_t>(std::size(symbols))}, at::kFloat);
        std::vector<int64_t> major(n);
        for (int64_t i = 0; i < n; ++i) {
            major[i] = i;
            logits.index_put_({i, static_cast<int64_t>(symbols.find(cons[i]))}, 1.0f);
        }
        return VariantCallingSample{.seq_id = 0,
                                    .positions_major = major,
                                    .positions_minor = std::vector<int64_t>(n, 0),
                                    .logits = logits,
                                    .depth = at::ones({n}, at::kInt)};
    };

    using Extents = std::vector<std::pair<int64_t, int64_t>>;

    const auto split = [&](const std::string_view cons, const std::vector<int64_t>& boundaries) {
        std::vector<VariantCallingSample> samples;
        samples.emplace_back(make_sample(cons));
        const auto ret =
                split_vc_samples_on_boundaries(std::move(samples), boundaries, draft, decoder);
        Extents extents;
        extents.reserve(std::size(ret));
        for (const auto& vc : ret) {
            extents.emplace_back(vc.start(), vc.end());
        }
        return extents;
    };

    CATCH_SECTION("No boundaries") { CATCH_CHECK(split(draft, {}) == Extents{{0, 10}}); }
    CATCH_SECTION("Boundaries at or outside the sample edges are ignored") {
        CATCH_CHECK(split(draft, {0, 10, 20}) == Extents{{0, 10}});
    }
    CATCH_SECTION("Boundary on a non-variant column, one split") {
        CATCH_CHECK(split(draft, {5}) == Extents{{0, 5}, {5, 10}});
    }
    CATCH_SECTION("Boundary on a non-variant column, two splits") {
        CATCH_CHECK(split(draft, {3, 7}) == Extents{{0, 3}, {3, 7}, {7, 10}});
    }
    CATCH_SECTION("Boundary inside a variant is deferred to the variant end") {
        // Draft[4:6] = "AC", consensus "CA".
        CATCH_CHECK(split("ACGTCAGTAC", {5}) == Extents{{0, 6}, {6, 10}});
    }
    CATCH_SECTION("Multiple boundaries inside a variant coincide") {
        CATCH_CHECK(split("ACGTCAGTAC", {5, 6}) == Extents{{0, 6}, {6, 10}});
    }
    CATCH_SECTION("Boundary adjacent to variant start") {
        CATCH_CHECK(split("ACGTCAGTAC", {4}) == Extents{{0, 4}, {4, 10}});
    }
    CATCH_SECTION("Boundary adjacent to variant end") {
        CATCH_CHECK(split("ACGTCAGTAC", {6}) == Extents{{0, 6}, {6, 10}});
    }
    CATCH_SECTION("Variant reaching the sample start is ignored") {
        // Draft[0:3] = "ACG", consensus "CAT".
        CATCH_CHECK(split("CATTACGTAC", {2}) == Extents{{0, 10}});
    }
    CATCH_SECTION("Variant reaching the sample end is ignored") {
        // Draft[8:10] = "AC", consensus "CA".
        CATCH_CHECK(split("ACGTACGTCA", {9}) == Extents{{0, 10}});
    }
}

}  // namespace dorado::secondary::consensus::tests
