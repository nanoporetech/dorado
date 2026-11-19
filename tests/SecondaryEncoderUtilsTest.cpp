#include "secondary/features/encoder_utils.h"
#include "secondary/features/medaka_read_matrix_constants.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <torch/types.h>

#include <cstdint>
#include <string>
#include <vector>

#define TEST_GROUP "[SecondaryEncoderUtils]"

namespace dorado::secondary::tests {

CATCH_TEST_CASE("plan_chunk_reorder", TEST_GROUP) {
    const std::string blank(BLANK_READ_PLACEHOLDER);
    struct TestCase {
        std::string test_name;
        std::vector<std::string> prev_rids_out;
        std::vector<std::string> rids_in;
        std::vector<std::string> rids_out;
        bool expected_requires_reorder;
        std::vector<int64_t> expected_rows;
        std::vector<std::string> expected_out_ids;
    };

    auto [test_case] = GENERATE_REF(table<TestCase>({
            TestCase{"Empty input, returns empty plan", {}, {}, {}, false, {}, {}},

            TestCase{
                    "Equal IDs, rows stay in place",
                    {"A", "B"},
                    {"A", "B"},
                    {"oA", "oB"},
                    false,
                    {0, 1},
                    {"oA", "oB"},
            },

            TestCase{
                    "Valid order, previous chunk is deeper, rows are added",
                    {"A", "B", "C"},
                    {"A", blank},
                    {"oA", "oBlank"},
                    false,
                    {0, 1, -1},
                    {"oA", "oBlank", blank},
            },

            TestCase{
                    "Valid order, this chunk is deeper",
                    {"A"},
                    {"A", "B"},
                    {"oA", "oB"},
                    false,
                    {0, 1},
                    {"oA", "oB"},
            },

            TestCase{
                    "Rows are permuted and an empty row is left for a read which ended",
                    {"A", "B", "C"},
                    {"C", "A"},
                    {"oC", "oA"},
                    true,
                    {1, -1, 0},
                    {"oA", blank, "oC"},
            },

            TestCase{
                    "Different named reads are not paired, rows are added",
                    {"A", "B"},
                    {"C", "D"},
                    {"oC", "oD"},
                    true,
                    {-1, -1, 0, 1},
                    {blank, blank, "oC", "oD"},
            },

            TestCase{
                    "Two reads from prev_rids_out no longer exist and there is one added in "
                    "rids_in. Also, in the chunk, read ID in row 3 is replaced from 'D' to 'H'.",
                    {"A", "B", "C", "D", "E", "G"},
                    {"A", "C", "D", "F", "E"},
                    {"A", "C", "H", "F", "E"},
                    true,
                    {0, -1, 1, 2, 4, -1, 3},
                    {"A", blank, "C", "H", "E", blank, "F"},
            },

            TestCase{
                    "Unmatched in rows can be matched with empty previous out rows",
                    {"A", blank, "C", "D", "E", "G"},
                    {"A", "C", "D", "F", "E"},
                    {"A", "C", "H", "F", "E"},
                    true,
                    {0, 3, 1, 2, 4, -1},
                    {"A", "F", "C", "H", "E", blank},
            },

            TestCase{
                    "Empty in rows can be matched with unmatched out rows",
                    {"A", "B", "C", "D", "E", "G"},
                    {"A", "C", "D", blank, "E"},
                    {"A", "C", "H", blank, "E"},
                    true,
                    {0, 3, 1, 2, 4, -1},
                    {"A", blank, "C", "H", "E", blank},
            },

            TestCase{
                    "Blanks can be matched with blanks",
                    {"A", blank, "C", "D", "E", blank},
                    {"A", "C", "D", blank, "E"},
                    {"A", "C", "H", blank, "E"},
                    true,
                    {0, 3, 1, 2, 4, -1},
                    {"A", blank, "C", "H", "E", blank},
            },

            TestCase{
                    "Blank IDs are not matched by name, even if prev_rids_out and rids_in share a "
                    "blank ID.",
                    {blank, "B"},
                    {"N", blank},
                    {"oN", "oBlank"},
                    false,
                    {0, 1},
                    {"oN", "oBlank"},
            },

            TestCase{
                    "Pairing does not depend on the order of the blank rows.",
                    {"A", blank, "B"},
                    {blank, "A", "N"},
                    {"oBlank", "oA", "oN"},
                    true,
                    {1, 2, 0},
                    {"oA", "oN", "oBlank"},
            },

            TestCase{
                    "No reordering. Valid order is kept when IDs are paired by index with at least "
                    "one blank.",
                    {blank, blank},
                    {blank, "N"},
                    {"oBlank", "oN"},
                    false,
                    {0, 1},
                    {"oBlank", "oN"},
            },

            TestCase{
                    "No reordering. All rows are blank, match with any input,"
                    "one blank.",
                    {"A", blank, "B"},
                    {blank, blank},
                    {blank, blank},
                    false,
                    {0, 1, -1},
                    {blank, blank, blank},
            },

            TestCase{
                    "No reordering. Valid order is kept and the chunk is padded when prev_rids_out "
                    "is longer.",
                    {blank, blank, "C"},
                    {"N", blank},
                    {"oN", "oBlank"},
                    false,
                    {0, 1, -1},
                    {"oN", "oBlank", blank},
            },

            TestCase{
                    "Reordering. A common ID at a different index requires reordering even if all "
                    "rows pair by index.",
                    {"A", blank},
                    {blank, "A"},
                    {"oBlank", "oA"},
                    true,
                    {1, 0},
                    {"oA", "oBlank"},
            },

            TestCase{
                    "Reordering. Duplicate read IDs cause the final chunk to have more rows than "
                    "before.",
                    {"A", "C", "C", "C", "D", "E"},
                    {"A", "C", "D", "E", "X", "X"},
                    {"A", "C", "H", "E", "X", "X"},
                    true,
                    {0, 1, 1, 1, 2, 3, 4, 5},
                    {"A", "C", "C", "C", "H", "E", "X", "X"},
            },

            TestCase{
                    "No reordering. Preceding chunk is empty.",
                    {},
                    {"A", "B"},
                    {"A", "B"},
                    false,
                    {0, 1},
                    {"A", "B"},
            },

            TestCase{
                    "No reordering. This chunk is empty, padded to depth of preceding chunk.",
                    {"A", blank, "B"},
                    {},
                    {},
                    false,
                    {-1, -1, -1},
                    {blank, blank, blank},
            },
    }));

    CATCH_INFO(TEST_GROUP << " Test name: " << test_case.test_name);

    const ChunkReorderPlan plan =
            plan_chunk_reorder(test_case.prev_rids_out, test_case.rids_in, test_case.rids_out);
    CATCH_CHECK(plan.requires_reorder == test_case.expected_requires_reorder);
    CATCH_CHECK(plan.src_rows == test_case.expected_rows);
    CATCH_CHECK(plan.next_rids_out == test_case.expected_out_ids);
}

CATCH_TEST_CASE("merge_read_chunks", TEST_GROUP) {
    const std::string blank(BLANK_READ_PLACEHOLDER);
    const auto make_tensor = [](const std::vector<int8_t>& values,
                                const std::vector<int64_t>& shape) {
        return at::tensor(values, at::kChar).reshape(shape);
    };

    struct TestCase {
        std::string test_name;
        std::vector<at::Tensor> chunks;
        std::vector<std::vector<std::string>> left_rids;
        std::vector<std::vector<std::string>> right_rids;
        bool should_throw{false};
        at::Tensor expected_chunk;
        std::vector<std::string> expected_left_rids;
        std::vector<std::string> expected_right_rids;
    };

    auto [test_case] = GENERATE_REF(table<TestCase>({
            TestCase{"Empty input returns empty", {}, {}, {}, false, {}, {}, {}},
            TestCase{
                    "Single chunk is returned unchanged",
                    {make_tensor({0, 1, 2, 3}, {2, 2})},
                    {{"l0", "l1"}},
                    {{"r0", "r1"}},
                    false,
                    make_tensor({0, 1, 2, 3}, {2, 2}),
                    {"l0", "l1"},
                    {"r0", "r1"},
            },

            TestCase{
                    "Same depth, no padding, equals a concatenation",
                    {make_tensor({0, 1, 2, 3}, {2, 2}), make_tensor({4, 5, 6, 7}, {2, 2})},
                    {{"l0", "l1"}, {"a", "b"}},
                    {{"a", "b"}, {"r0", "r1"}},
                    false,
                    make_tensor({0, 1, 2, 3, 4, 5, 6, 7}, {4, 2}),
                    {"l0", "l1"},
                    {"r0", "r1"},
            },

            TestCase{
                    "Depth grows 2 -> 3 -> 4, padded once",
                    {make_tensor({1, 2}, {1, 2}), make_tensor({3, 4, 5}, {1, 3}),
                     make_tensor({6, 7, 8, 9}, {1, 4})},
                    {{"l0", "l1"}, {"a", "b", "c"}, {"d", "e", "f", "g"}},
                    {{"a", "b"}, {"d", "e", "f"}, {"h", "i", "j", "k"}},
                    false,
                    make_tensor({1, 2, 0, 0, 3, 4, 5, 0, 6, 7, 8, 9}, {3, 4}),
                    {"l0", "l1", blank, blank},
                    {"h", "i", "j", "k"},
            },

            TestCase{
                    "Depth grows then shrinks 3 -> 4 -> 2, padded once",
                    {make_tensor({1, 2, 3}, {1, 3}), make_tensor({4, 5, 6, 7}, {1, 4}),
                     make_tensor({8, 9}, {1, 2})},
                    {{"l0", "l1", "l2"}, {"a", "b", "c", "d"}, {"e", "f"}},
                    {{"a", "b", "c"}, {"e", "f", "g", "h"}, {"x", "y"}},
                    false,
                    make_tensor({1, 2, 3, 0, 4, 5, 6, 7, 8, 9, 0, 0}, {3, 4}),
                    {"l0", "l1", "l2", blank},
                    {"x", "y", blank, blank},
            },

            // Second chunk has reads (b, N, a) while the first chunk ended with reads (a, b).
            // Rows of the second chunk are placed as (a, b, N). Chunks are [1 x reads x 2].
            TestCase{
                    "Permuted chunk with padding, 3D chunks with 2 features",
                    {make_tensor({1, 2, 3, 4}, {1, 2, 2}),
                     make_tensor({10, 11, 20, 21, 30, 31}, {1, 3, 2})},
                    {{"l0", "l1"}, {"b", "N", "a"}},
                    {{"a", "b"}, {"ob", "oN", "oa"}},
                    false,
                    make_tensor({1, 2, 3, 4, 0, 0, 30, 31, 10, 11, 20, 21}, {2, 3, 2}),
                    {"l0", "l1", blank},
                    {"oa", "ob", "oN"},
            },

            // Chunks of 2, 3 and 1 positions. The second chunk is permuted (b, N, a) -> (a, b, N)
            // and the first is padded, with the third chunk following at offset 2 + 3.
            TestCase{
                    "Chunks with different numbers of positions, permuted and padded",
                    {make_tensor({1, 2, 3, 4}, {2, 2}),
                     make_tensor({10, 11, 12, 13, 14, 15, 16, 17, 18}, {3, 3}),
                     make_tensor({20, 21, 22}, {1, 3})},
                    {{"l0", "l1"}, {"b", "N", "a"}, {"oa", "ob", "oN"}},
                    {{"a", "b"}, {"ob", "oN", "oa"}, {"x", "y", "z"}},
                    false,
                    make_tensor({1, 2, 0, 3, 4, 0, 12, 10, 11, 15, 13, 14, 18, 16, 17, 20, 21, 22},
                                {6, 3}),
                    {"l0", "l1", blank},
                    {"x", "y", "z"},
            },

            // The third chunk is reordered against the right IDs of the reordered, padded second
            // chunk (oa, ob, oN), which adds a row. The depth is 4 for all chunks in the end.
            TestCase{
                    "Three chunks, the third is reordered against the reordered second",
                    {make_tensor({1, 2}, {1, 2}), make_tensor({3, 4, 5}, {1, 3}),
                     make_tensor({6, 7, 8}, {1, 3})},
                    {{"l0", "l1"}, {"b", "N", "a"}, {"oN", "oa", "Z"}},
                    {{"a", "b"}, {"ob", "oN", "oa"}, {"pN", "pa", "pZ"}},
                    false,
                    make_tensor({1, 2, 0, 0, 5, 3, 4, 0, 7, 0, 6, 8}, {3, 4}),
                    {"l0", "l1", blank, blank},
                    {"pa", blank, "pN", "pZ"},
            },

            // The first chunk is the deepest, so the left IDs need no padding and the shallower
            // second chunk is padded.
            TestCase{
                    "First chunk is the deepest, later chunk is padded",
                    {make_tensor({1, 2, 3, 4}, {1, 4}), make_tensor({5, 6}, {1, 2})},
                    {{"l0", "l1", "l2", "l3"}, {"a", "b"}},
                    {{"a", "b", "c", "d"}, {"e", "f"}},
                    false,
                    make_tensor({1, 2, 3, 4, 5, 6, 0, 0}, {2, 4}),
                    {"l0", "l1", "l2", "l3"},
                    {"e", "f", blank, blank},
            },

            // Chunks which are [positions x reads x 2] views of larger tensors, sliced on the reads
            // dimension as produced by the read matrix tensors. The second chunk is permuted.
            TestCase{
                    "Non-contiguous 3D chunks, permuted and padded",
                    {make_tensor({1, 2, 3, 4, 9, 9, 9, 9, 5, 6, 7, 8, 9, 9, 9, 9}, {2, 4, 2})
                             .slice(1, 0, 2),
                     make_tensor({10, 11, 20, 21, 30, 31, 99, 99, 99, 99,
                                  12, 13, 22, 23, 32, 33, 99, 99, 99, 99},
                                 {2, 5, 2})
                             .slice(1, 0, 3)},
                    {{"l0", "l1"}, {"b", "N", "a"}},
                    {{"a", "b"}, {"ob", "oN", "oa"}},
                    false,
                    make_tensor({1,  2,  3,  4,  0,  0,  5,  6,  7,  8,  0,  0,
                                 30, 31, 10, 11, 20, 21, 32, 33, 12, 13, 22, 23},
                                {4, 3, 2}),
                    {"l0", "l1", blank},
                    {"oa", "ob", "oN"},
            },

            TestCase{
                    "Chunk with zero depth first, the other chunk is padded",
                    {make_tensor({}, {1, 0}), make_tensor({5, 6}, {1, 2})},
                    {{}, {"a", "b"}},
                    {{}, {"c", "d"}},
                    false,
                    make_tensor({0, 0, 5, 6}, {2, 2}),
                    {blank, blank},
                    {"c", "d"},
            },

            TestCase{
                    "Chunk with zero depth last is padded",
                    {make_tensor({1, 2}, {1, 2}), make_tensor({}, {1, 0})},
                    {{"a", "b"}, {}},
                    {{"c", "d"}, {}},
                    false,
                    make_tensor({1, 2, 0, 0}, {2, 2}),
                    {"a", "b"},
                    {blank, blank},
            },

            TestCase{
                    "Chunk with zero positions adds no rows",
                    {make_tensor({1, 2}, {1, 2}), make_tensor({}, {0, 2})},
                    {{"l0", "l1"}, {"a", "b"}},
                    {{"a", "b"}, {"c", "d"}},
                    false,
                    make_tensor({1, 2}, {1, 2}),
                    {"l0", "l1"},
                    {"c", "d"},
            },

            TestCase{
                    "Undefined first chunk throws",
                    {at::Tensor{}, make_tensor({1, 2}, {1, 2})},
                    {{"l0", "l1"}, {"a", "b"}},
                    {{"a", "b"}, {"d", "e"}},
                    true,
                    {},
                    {},
                    {},
            },

            TestCase{
                    "Undefined chunks throw",
                    {make_tensor({1, 2}, {1, 2}), at::Tensor{}},
                    {{"l0", "l1"}, {"a", "b"}},
                    {{"a", "b"}, {"d", "e"}},
                    true,
                    {},
                    {},
                    {},
            },

            TestCase{
                    "Chunks which are not at least 2D throw",
                    {make_tensor({1, 2}, {1, 2}), make_tensor({3, 4}, {2})},
                    {{"l0", "l1"}, {"a", "b"}},
                    {{"a", "b"}, {"d", "e"}},
                    true,
                    {},
                    {},
                    {},
            },

            TestCase{
                    "Chunks with different inner shape throw",
                    {make_tensor({1, 2}, {1, 2}), make_tensor({3, 4}, {1, 2, 1})},
                    {{"l0", "l1"}, {"a", "b"}},
                    {{"a", "b"}, {"d", "e"}},
                    true,
                    {},
                    {},
                    {},
            },

            TestCase{
                    "3D chunks with different feature sizes throw",
                    {make_tensor({1, 2, 3, 4}, {1, 2, 2}),
                     make_tensor({1, 2, 3, 4, 5, 6}, {1, 2, 3})},
                    {{"l0", "l1"}, {"a", "b"}},
                    {{"a", "b"}, {"d", "e"}},
                    true,
                    {},
                    {},
                    {},
            },

            TestCase{
                    "Chunks with different dtype throw",
                    {make_tensor({1, 2}, {1, 2}), make_tensor({3, 4}, {1, 2}).to(at::kInt)},
                    {{"l0", "l1"}, {"a", "b"}},
                    {{"a", "b"}, {"d", "e"}},
                    true,
                    {},
                    {},
                    {},
            },

            TestCase{
                    "Read IDs which do not match the depth of the chunk throw",
                    {make_tensor({1, 2}, {1, 2}), make_tensor({3, 4}, {1, 2})},
                    {{"l0", "l1"}, {"a", "b", "c"}},
                    {{"a", "b"}, {"d", "e"}},
                    true,
                    {},
                    {},
                    {},
            },
    }));

    CATCH_INFO(TEST_GROUP << " Test name: " << test_case.test_name);

    if (test_case.should_throw) {
        CATCH_CHECK_THROWS_AS(
                merge_read_chunks(test_case.chunks, test_case.left_rids, test_case.right_rids),
                std::runtime_error);
    } else {
        const auto [merged_chunk, merged_read_ids_left, merged_read_ids_right] =
                merge_read_chunks(test_case.chunks, test_case.left_rids, test_case.right_rids);
        if (test_case.expected_chunk.defined()) {
            CATCH_CHECK(merged_chunk.equal(test_case.expected_chunk));
        } else {
            CATCH_CHECK_FALSE(merged_chunk.defined());
        }
        CATCH_CHECK(merged_read_ids_left == test_case.expected_left_rids);
        CATCH_CHECK(merged_read_ids_right == test_case.expected_right_rids);
        // Inputs are released once copied.
        for (const auto& chunk : test_case.chunks) {
            CATCH_CHECK_FALSE(chunk.defined());
        }
    }
}

CATCH_TEST_CASE("filter_empty_minor_columns-01-normal_input_nothing_to_filter", TEST_GROUP) {
    /**
     * \brief Should remove the empty minor column (2, 1) and relabel the remaining minor columns for the same major
     *          to remove the gap.
    */
    // clang-format off
    // Features tensor shape: [pos, coverage, features]
    // Feature column: [base, qual, strand, mapq, dwell, haplotag, snp_qv, [dtype]
    const at::Tensor features = torch::tensor(
        {
            // (0,.,.)  Position: (0, 0)
            {{1, 0, 1, 51, 0, 0, 60},       // read_01
             {1, 0, 0, 52, 0, 3, 7},        // read_02
             {1, 0, 1, 53, 4, 5, 2},        // read_03
            },

            // (1,.,.)  Position: (1, 0)
            {{2, 0,  1, 51, 0, 0, 60},
             {2, 0,  0, 52, 0, 3,  7},
             {2, 0,  1, 53, 5, 5,  2},
            },

            // (2,.,.)  Position: (2, 0)
            {{4, 0,  1, 51, 0, 0, 60},
             {4, 0,  0, 52, 0, 3, 7},
             {4, 0,  1, 53, 2, 5, 2},
            },

            // (3,.,.)  Position: (2, 1)
            {{3, 0,  1, 51, 0, 0, 60},
             {3, 0,  0, 52, 0, 3,  7},
             {3, 0,  0,  0, 0, 0,  0}       // read_03 exits
            },

            // (4,.,.)  Position: (2, 2)
            {{1,  0,  1, 51, 0, 0, 60},
             {1,  0,  0, 52, 0, 3,  7},
             {0,  0,  0,  0, 0, 0,  0},
            },

            // (5,.,.)  Position: (3, 0)
            {{1, 0,  1, 51, 0, 0, 60},
             {0, 0,  0,  0, 0, 0,  0},      // read_02 exits
             {0, 0,  0,  0, 0, 0,  0},
            },

            // (6,.,.)  Position: (4, 0)
            {{2, 0,  1, 51, 0, 0, 60},
             {0, 0,  0,  0, 0, 0,  0},
             {0, 0,  0,  0, 0, 0,  0},
            },

        }, torch::dtype(torch::kInt8)
    );
    const std::vector<int64_t> positions_major {
        0, 1, 2, 2, 2, 3, 4,
    };
    const std::vector<int64_t> positions_minor {
        0, 0, 0, 1, 2, 0, 0,
    };
    // clang-format on

    const auto [result_features, result_positions_major, result_positions_minor] =
            filter_empty_minor_columns(features, positions_major, positions_minor);

    CATCH_CHECK(result_features.equal(features));
    CATCH_CHECK(result_positions_major == positions_major);
    CATCH_CHECK(result_positions_minor == positions_minor);
}

CATCH_TEST_CASE("filter_empty_minor_columns-02-empty_input_throws", TEST_GROUP) {
    /**
     * \brief Empty input tensor should throw.
    */
    const at::Tensor features = torch::tensor({}, torch::dtype(torch::kInt8));
    const std::vector<int64_t> positions_major{};
    const std::vector<int64_t> positions_minor{};

    CATCH_CHECK_THROWS(filter_empty_minor_columns(features, positions_major, positions_minor));
}

CATCH_TEST_CASE("filter_empty_minor_columns-03-uninitialized_input_tensor", TEST_GROUP) {
    /**
     * \brief Uninitialized input tensor should throw.
    */
    const at::Tensor features;
    const std::vector<int64_t> positions_major{};
    const std::vector<int64_t> positions_minor{};

    CATCH_CHECK_THROWS(filter_empty_minor_columns(features, positions_major, positions_minor));
}

CATCH_TEST_CASE("filter_empty_minor_columns-04-position_length_mismatch", TEST_GROUP) {
    /**
     * \brief Mismatch in length of the number of positions in the input tensor and the lengths of the
     *          positions_major, positions_minor.
    */
    // clang-format off
    // Features tensor shape: [pos, coverage, features]
    // Feature column: [base, qual, strand, mapq, dwell, haplotag, snp_qv, [dtype]
    const at::Tensor features = torch::tensor(
        {
            // (0,.,.)  Position: (0, 0)
            {{1, 0, 1, 51, 0, 0, 60},       // read_01
             {1, 0, 0, 52, 0, 3, 7},        // read_02
             {1, 0, 1, 53, 4, 5, 2},        // read_03
            },

            // (1,.,.)  Position: (1, 0)
            {{2, 0,  1, 51, 0, 0, 60},
             {2, 0,  0, 52, 0, 3,  7},
             {2, 0,  1, 53, 5, 5,  2},
            },
        }, torch::dtype(torch::kInt8)
    );
    // clang-format on

    CATCH_SECTION("normal case, everything matches, should pass") {
        // clang-format off
        const std::vector<int64_t> positions_major {
            0, 1,
        };
        const std::vector<int64_t> positions_minor {
            0, 0,
        };
        // clang-format on

        const auto [result_features, result_positions_major, result_positions_minor] =
                filter_empty_minor_columns(features, positions_major, positions_minor);

        CATCH_CHECK(result_features.equal(features));
        CATCH_CHECK(result_positions_major == positions_major);
        CATCH_CHECK(result_positions_minor == positions_minor);
    }

    CATCH_SECTION("positions_major are of wrong length") {
        // clang-format off
        const std::vector<int64_t> positions_major {
            0,
        };
        const std::vector<int64_t> positions_minor {
            0, 0,
        };
        // clang-format on

        CATCH_CHECK_THROWS(filter_empty_minor_columns(features, positions_major, positions_minor));
    }

    CATCH_SECTION("positions_minor are of wrong length") {
        // clang-format off
        const std::vector<int64_t> positions_major {
            0, 1,
        };
        const std::vector<int64_t> positions_minor {
            0,
        };
        // clang-format on

        CATCH_CHECK_THROWS(filter_empty_minor_columns(features, positions_major, positions_minor));
    }
}

CATCH_TEST_CASE("filter_empty_minor_columns-05-minor_internal_left_aligned", TEST_GROUP) {
    /**
     * \brief Should remove the empty minor column (2, 1) and relabel the remaining minor columns for the same major
     *          to remove the gap.
    */
    // clang-format off
    // Features tensor shape: [pos, coverage, features]
    // Feature column: [base, qual, strand, mapq, dwell, haplotag, snp_qv, [dtype]
    const at::Tensor features = torch::tensor(
        {
            // (0,.,.)  Position: (0, 0)
            {{1, 0, 1, 51, 0, 0, 60},       // read_01
             {1, 0, 0, 52, 0, 3, 7},        // read_02
             {1, 0, 1, 53, 4, 5, 2},        // read_03
            },

            // (1,.,.)  Position: (1, 0)
            {{2, 0,  1, 51, 0, 0, 60},
             {2, 0,  0, 52, 0, 3,  7},
             {2, 0,  1, 53, 5, 5,  2},
            },

            // (2,.,.)  Position: (2, 0)
            {{4, 0,  1, 51, 0, 0, 60},
             {4, 0,  0, 52, 0, 3, 7},
             {4, 0,  1, 53, 2, 5, 2},
            },

            // (3,.,.)  Position: (2, 1)
            {{0, 0,  1, 51, 0, 0, 60},
             {0, 0,  0, 52, 0, 3,  7},
             {0, 0,  0,  0, 0, 0,  0}       // read_03 exits
            },

            // (4,.,.)  Position: (2, 2)
            {{1,  0,  1, 51, 0, 0, 60},
             {1,  0,  0, 52, 0, 3,  7},
             {0,  0,  0,  0, 0, 0,  0},
            },

            // (5,.,.)  Position: (3, 0)
            {{1, 0,  1, 51, 0, 0, 60},
             {0, 0,  0,  0, 0, 0,  0},      // read_02 exits
             {0, 0,  0,  0, 0, 0,  0},
            },

            // (6,.,.)  Position: (4, 0)
            {{2, 0,  1, 51, 0, 0, 60},
             {0, 0,  0,  0, 0, 0,  0},
             {0, 0,  0,  0, 0, 0,  0},
            },

        }, torch::dtype(torch::kInt8)
    );
    const std::vector<int64_t> positions_major {
        0, 1, 2, 2, 2, 3, 4,
    };
    const std::vector<int64_t> positions_minor {
        0, 0, 0, 1, 2, 0, 0,
    };
    // clang-format on

    // clang-format off
    const at::Tensor expected_features = torch::tensor(
        {
            // (0,.,.)  Position: (0, 0)
            {{1, 0, 1, 51, 0, 0, 60},           // read_01
             {1, 0, 0, 52, 0, 3, 7},            // read_02
             {1, 0, 1, 53, 4, 5, 2},            // read_03
            },

            // (1,.,.)  Position: (1, 0)
            {{2, 0,  1, 51, 0, 0, 60},
             {2, 0,  0, 52, 0, 3,  7},
             {2, 0,  1, 53, 5, 5,  2},
            },

            // (2,.,.)  Position: (2, 0)
            {{4, 0,  1, 51, 0, 0, 60},
             {4, 0,  0, 52, 0, 3, 7},
             {4, 0,  1, 53, 2, 5, 2},
            },

            // // (3,.,.)  Position: (2, 1)
            // {{0, 0,  1, 51, 0, 0, 60},
            //  {0, 0,  0, 52, 0, 3,  7},
            //  {0, 0,  0,  0, 0, 0,  0}        // read_03 exits
            // },

            // (3,.,.)  Position: (2, 2)
            {{1,  0,  1, 51, 0, 0, 60},
             {1,  0,  0, 52, 0, 3,  7},
             {0,  0,  0,  0, 0, 0,  0},
            },

            // (4,.,.)  Position: (3, 0)
            {{1, 0,  1, 51, 0, 0, 60},
             {0, 0,  0,  0, 0, 0,  0},          // read_02 exits
             {0, 0,  0,  0, 0, 0,  0},
            },

            // (5,.,.)  Position: (4, 0)
            {{2, 0,  1, 51, 0, 0, 60},
             {0, 0,  0,  0, 0, 0,  0},
             {0, 0,  0,  0, 0, 0,  0},
            },

        }, torch::dtype(torch::kInt8)
    );
    const std::vector<int64_t> expected_positions_major {
        0, 1, 2, 2, 3, 4,
    };
    const std::vector<int64_t> expected_positions_minor {
        0, 0, 0, 1, 0, 0,
    };
    // clang-format on

    const auto [result_features, result_positions_major, result_positions_minor] =
            filter_empty_minor_columns(features, positions_major, positions_minor);

    CATCH_CHECK(result_features.equal(expected_features));
    CATCH_CHECK(result_positions_major == expected_positions_major);
    CATCH_CHECK(result_positions_minor == expected_positions_minor);
}

CATCH_TEST_CASE("filter_empty_minor_columns-06-minor_first_position", TEST_GROUP) {
    /**
     * \brief Should remove the minor column at the very beginning of the window.
     *          Window begins in the middle of a minor position stretch (major = 5, minor = 2) and this column should be removed
     *          because all bases are equal to 0.
     *          When the first column is removed, the minor positions need to be relabeled, otherwise the window will
     *          begin on a different minor and may potentially not be merged with the previous window.
     */
    // clang-format off
    // Features tensor shape: [pos, coverage, features]
    // Feature column: [base, qual, strand, mapq, dwell, haplotag, snp_qv, [dtype]
    const at::Tensor features = torch::tensor(
        {
            // (0,.,.)  Position: (5, 2)    <- Feature tensor begins on a minor position
            {{0, 0, 1, 51, 0, 0, 60},
             {0, 0, 0, 52, 0, 3, 7},
             {0, 0, 1, 53, 4, 5, 2},
            },

            // (1,.,.)  Position: (5, 3)
            {{2, 0,  1, 51, 0, 0, 60},
             {2, 0,  0, 52, 0, 3,  7},
             {2, 0,  1, 53, 5, 5,  2},
            },

            // (2,.,.)  Position: (5, 4)
            {{4, 0,  1, 51, 0, 0, 60},
             {4, 0,  0, 52, 0, 3, 7},
             {4, 0,  1, 53, 2, 5, 2},
            },

            // (4,.,.)  Position: (6, 0)
            {{1,  0,  1, 51, 0, 0, 60},
             {1,  0,  0, 52, 0, 3,  7},
             {0,  0,  0,  0, 0, 0,  0},
            },

            // (5,.,.)  Position: (7, 0)
            {{1, 0,  1, 51, 0, 0, 60},
             {0, 0,  0,  0, 0, 0,  0},
             {0, 0,  0,  0, 0, 0,  0},
            },

            // (6,.,.)  Position: (8, 0)
            {{2, 0,  1, 51, 0, 0, 60},
             {0, 0,  0,  0, 0, 0,  0},
             {0, 0,  0,  0, 0, 0,  0},
            },
        }, torch::dtype(torch::kInt8)
    );
    const std::vector<int64_t> positions_major {
        5, 5, 5, 6, 7, 8,
    };
    const std::vector<int64_t> positions_minor {
        2, 3, 4, 0, 0, 0,
    };
    // clang-format on

    // clang-format off
    const at::Tensor expected_features = torch::tensor(
        {
            // // (0,.,.)  Position: (5, 2) <- Feature tensor begins on a minor position
            // {{1, 0, 1, 51, 0, 0, 60},
            //  {1, 0, 0, 52, 0, 3, 7},
            //  {1, 0, 1, 53, 4, 5, 2},
            // },

            // (0,.,.)  Position: (5, 2)    <- Feature tensor begins on a minor position
            {{2, 0,  1, 51, 0, 0, 60},
             {2, 0,  0, 52, 0, 3,  7},
             {2, 0,  1, 53, 5, 5,  2},
            },

            // (1,.,.)  Position: (5, 3)
            {{4, 0,  1, 51, 0, 0, 60},
             {4, 0,  0, 52, 0, 3, 7},
             {4, 0,  1, 53, 2, 5, 2},
            },

            // (2,.,.)  Position: (6, 0)
            {{1,  0,  1, 51, 0, 0, 60},
             {1,  0,  0, 52, 0, 3,  7},
             {0,  0,  0,  0, 0, 0,  0},
            },

            // (3,.,.)  Position: (7, 0)
            {{1, 0,  1, 51, 0, 0, 60},
             {0, 0,  0,  0, 0, 0,  0},
             {0, 0,  0,  0, 0, 0,  0},
            },

            // (4,.,.)  Position: (8, 0)
            {{2, 0,  1, 51, 0, 0, 60},
             {0, 0,  0,  0, 0, 0,  0},
             {0, 0,  0,  0, 0, 0,  0},
            },

        }, torch::dtype(torch::kInt8)
    );
    const std::vector<int64_t> expected_positions_major {
        5, 5, 6, 7, 8,
    };
    const std::vector<int64_t> expected_positions_minor {
        2, 3, 0, 0, 0,
    };
    // clang-format on

    const auto [result_features, result_positions_major, result_positions_minor] =
            filter_empty_minor_columns(features, positions_major, positions_minor);

    CATCH_CHECK(result_features.equal(expected_features));
    CATCH_CHECK(result_positions_major == expected_positions_major);
    CATCH_CHECK(result_positions_minor == expected_positions_minor);
}

CATCH_TEST_CASE("filter_empty_minor_columns-07-multiple_minor_first_positions", TEST_GROUP) {
    /**
     * \brief Should remove all minor columns at the very beginning of the window.
     *          After this, the window should now begin with another major and should not be relabeled.
     *          This is a test for an edge case.
     *          Window begins in the middle of a minor position stretch (major = 5, minor = 2).
     */
    // clang-format off
    // Features tensor shape: [pos, coverage, features]
    // Feature column: [base, qual, strand, mapq, dwell, haplotag, snp_qv, [dtype]
    const at::Tensor features = torch::tensor(
        {
            // (0,.,.)  Position: (5, 2)    <- Feature tensor begins on a minor position
            {{0, 0, 1, 51, 0, 0, 60},
             {0, 0, 0, 52, 0, 3, 7},
             {0, 0, 1, 53, 4, 5, 2},
            },

            // (1,.,.)  Position: (5, 3)
            {{0, 0,  1, 51, 0, 0, 60},
             {0, 0,  0, 52, 0, 3,  7},
             {0, 0,  1, 53, 5, 5,  2},
            },

            // (2,.,.)  Position: (5, 4)
            {{0, 0,  1, 51, 0, 0, 60},
             {0, 0,  0, 52, 0, 3, 7},
             {0, 0,  1, 53, 2, 5, 2},
            },

            // (4,.,.)  Position: (6, 0)
            {{1,  0,  1, 51, 0, 0, 60},
             {1,  0,  0, 52, 0, 3,  7},
             {0,  0,  0,  0, 0, 0,  0},
            },

            // (5,.,.)  Position: (7, 0)
            {{1, 0,  1, 51, 0, 0, 60},
             {0, 0,  0,  0, 0, 0,  0},
             {0, 0,  0,  0, 0, 0,  0},
            },

            // (6,.,.)  Position: (8, 0)
            {{2, 0,  1, 51, 0, 0, 60},
             {0, 0,  0,  0, 0, 0,  0},
             {0, 0,  0,  0, 0, 0,  0},
            },
        }, torch::dtype(torch::kInt8)
    );
    const std::vector<int64_t> positions_major {
        5, 5, 5, 6, 7, 8,
    };
    const std::vector<int64_t> positions_minor {
        2, 3, 4, 0, 0, 0,
    };
    // clang-format on

    // clang-format off
    const at::Tensor expected_features = torch::tensor(
        {
            // // (0,.,.)  Position: (5, 2)    <- Feature tensor begins on a minor position
            // {{0, 0, 1, 51, 0, 0, 60},
            //  {0, 0, 0, 52, 0, 3, 7},
            //  {0, 0, 1, 53, 4, 5, 2},
            // },

            // // (1,.,.)  Position: (5, 3)
            // {{0, 0,  1, 51, 0, 0, 60},
            //  {0, 0,  0, 52, 0, 3,  7},
            //  {0, 0,  1, 53, 5, 5,  2},
            // },

            // // (2,.,.)  Position: (5, 4)
            // {{0, 0,  1, 51, 0, 0, 60},
            //  {0, 0,  0, 52, 0, 3, 7},
            //  {0, 0,  1, 53, 2, 5, 2},
            // },

            // (4,.,.)  Position: (6, 0)
            {{1,  0,  1, 51, 0, 0, 60},
             {1,  0,  0, 52, 0, 3,  7},
             {0,  0,  0,  0, 0, 0,  0},
            },

            // (5,.,.)  Position: (7, 0)
            {{1, 0,  1, 51, 0, 0, 60},
             {0, 0,  0,  0, 0, 0,  0},
             {0, 0,  0,  0, 0, 0,  0},
            },

            // (6,.,.)  Position: (8, 0)
            {{2, 0,  1, 51, 0, 0, 60},
             {0, 0,  0,  0, 0, 0,  0},
             {0, 0,  0,  0, 0, 0,  0},
            },
        }, torch::dtype(torch::kInt8)
    );
    const std::vector<int64_t> expected_positions_major {
        6, 7, 8,
    };
    const std::vector<int64_t> expected_positions_minor {
        0, 0, 0,
    };
    // clang-format on

    const auto [result_features, result_positions_major, result_positions_minor] =
            filter_empty_minor_columns(features, positions_major, positions_minor);

    CATCH_CHECK(result_features.equal(expected_features));
    CATCH_CHECK(result_positions_major == expected_positions_major);
    CATCH_CHECK(result_positions_minor == expected_positions_minor);
}

}  // namespace dorado::secondary::tests
