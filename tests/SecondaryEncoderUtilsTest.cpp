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
    using Ids = std::vector<std::string>;
    using Rows = std::vector<int64_t>;

    CATCH_SECTION("Equal IDs, rows stay in place") {
        const ChunkReorderPlan plan = plan_chunk_reorder({"A", "B"}, {"A", "B"}, {"oA", "oB"});
        CATCH_CHECK_FALSE(plan.requires_reorder);
        CATCH_CHECK(plan.src_rows == Rows{0, 1});
        CATCH_CHECK(plan.next_rids_out == Ids{"oA", "oB"});
    }

    CATCH_SECTION("Valid order, previous chunk is deeper, rows are added") {
        const ChunkReorderPlan plan =
                plan_chunk_reorder({"A", "B", "C"}, {"A", blank}, {"oA", "oBlank"});
        CATCH_CHECK_FALSE(plan.requires_reorder);
        CATCH_CHECK(plan.src_rows == Rows{0, 1, -1});
        CATCH_CHECK(plan.next_rids_out == Ids{"oA", "oBlank", blank});
    }

    CATCH_SECTION("Valid order, this chunk is deeper") {
        const ChunkReorderPlan plan = plan_chunk_reorder({"A"}, {"A", "B"}, {"oA", "oB"});
        CATCH_CHECK_FALSE(plan.requires_reorder);
        CATCH_CHECK(plan.src_rows == Rows{0, 1});
        CATCH_CHECK(plan.next_rids_out == Ids{"oA", "oB"});
    }

    CATCH_SECTION("Rows are permuted and an empty row is left for a read which ended") {
        const ChunkReorderPlan plan = plan_chunk_reorder({"A", "B", "C"}, {"C", "A"}, {"oC", "oA"});
        CATCH_CHECK(plan.requires_reorder);
        CATCH_CHECK(plan.src_rows == Rows{1, -1, 0});
        CATCH_CHECK(plan.next_rids_out == Ids{"oA", blank, "oC"});
    }

    CATCH_SECTION("Different named reads are not paired, rows are added") {
        const ChunkReorderPlan plan = plan_chunk_reorder({"A", "B"}, {"C", "D"}, {"oC", "oD"});
        CATCH_CHECK(plan.requires_reorder);
        CATCH_CHECK(plan.src_rows == Rows{-1, -1, 0, 1});
        CATCH_CHECK(plan.next_rids_out == Ids{blank, blank, "oC", "oD"});
    }
}

CATCH_TEST_CASE("reorder_chunks", TEST_GROUP) {
    struct TestCase {
        std::string test_name;
        at::Tensor chunk;
        std::vector<std::string> prev_rids_out;
        std::vector<std::string> rids_in;
        std::vector<std::string> rids_out;
        at::Tensor expected_reordered_chunk;
        std::vector<std::string> expected_out_ids;
    };

    // clang-format off
    auto [test_case] = GENERATE_REF(table<TestCase>({
        TestCase{
            "Empty test",
            {}, {}, {}, {}, {}, {},
        },

        TestCase{
            "No reordering. Single row (dim = 1), input tensor is [1 x 1 x 4].",
            // chunk
            torch::tensor({
                {
                    {1, 1, 1, 1},    // row 1
                }
            }),
            // prev_rids_out
            {
                "read_0",
            },
            // rids_in
            {
                "read_0",
            },
            // rids_out
            {
                "read_0",
            },
            // expected_reordered_chunk
            torch::tensor({
                {
                    {1, 1, 1, 1},    // row 1
                }
            }),
            // expected_out_ids
            {
                "read_0",
            },
        },

        TestCase{
            "No reordering. Three rows (dim = 1), input tensor is [1 x 3 x 4].",
            // chunk
            torch::tensor({
                {
                    {1, 1, 1, 1},    // row 1
                    {2, 2, 2, 2},    // row 2
                    {3, 3, 3, 3},    // row 3
                }
            }),
            // prev_rids_out
            {
                "read_0",
                "read_1",
                "read_2",
            },
            // rids_in
            {
                "read_0",
                "read_1",
                "read_2",
            },
            // rids_out
            {
                "read_0",
                "read_1",
                "read_2",
            },
            // expected_reordered_chunk
            torch::tensor({
                {
                    {1, 1, 1, 1},    // row 1
                    {2, 2, 2, 2},    // row 2
                    {3, 3, 3, 3},    // row 3
                }
            }),
            // expected_out_ids
            {
                "read_0",
                "read_1",
                "read_2",
            },
        },

        /// Explanation for this test case:
        /// Previous chunk had 6 rows in the tensor, each row corresponding to a read name specified in:
        ///         prev_rids_out = {A, B, C, D, E, G}
        /// The new chunk has only 5 rows (so at least one read was dropped) but also has a slightly different set of reads:
        ///         rids_in = {A, C, D, F, E}
        /// The same new chunk also has a slightly different set of read IDs in the output, which means that within the pileup
        /// region of this chunk an alignment for one of the reads (D) has terminated and another read began (H) using the same
        /// row of the tensor:
        ///     1. Enumerate rids_in:
        ///         A = 0, C = 1, D = 2, F = 3, E = 4
        ///     2. Create new_indices and find matches of prev_rids_out in rids_in:
        ///             A -> 0, B -> -1, C -> 1, D -> 2, E -> 4, G -> -1
        ///             new_indices = {0, -1, 1, 2, 4, -1}
        ///     3. Determine the missing_in_indices (indices present in rids_in but not in prev_rids_out) and
        ///         missing_out_indices (indices present in prev_rids_out but not in rids_in)
        ///             missing_out_indices = {1, 5}
        ///             missing_in_indices = {3}
        ///     4. If any of the missing_out_indices indicate rows whose names indicate that they terminate with empty positions,
        ///         fill those positions from missing_in_indices. Alternatively, if any of the missing_in_indices correspond to
        ///         rows that start with empty positions, match these up with missing_out_indices.
        ///        In this case there are no such possible matches, but these are tested below.
        ///             new_indices = {0, -1, 1, 2, 4, -1}
        ///     5. If there are still any positions in missing_in_indices, append these to new_indices.
        ///             new_indices = {0, -1, 1, 2, 4, -1, 3}
        ///     6. Reorder chunk: permute with new_indices as the key, if new_indices[i] != -1.
        ///     7. Create thew next_rids_out. For new_indices[i] which exists, keep existing rids_out[i]. Otherwise,
        ///         set new_indices[i] = "__blank"
        ///
        ///         expected_out_ids = {A, __blank, C, H, E, __blank, F}
        ///
        TestCase{
            "Reordering. Five rows (dim = 1), input tensor is [1 x 5 x 4]. Two reads from prev_rids_out no "
            "longer exist and there is one added in rids_in. Also, in the chunk, read ID in row 3 is replaced from 'D' to 'H'.",
            // chunk
            torch::tensor({
                {
                    {1, 1, 1, 1},    // A
                    {2, 2, 2, 2},    // C
                    {3, 3, 3, 3},    // D -> H
                    {4, 4, 4, 4},    // F
                    {5, 5, 5, 5},    // E
                }
            }),
            // prev_rids_out
            {
                "A",
                "B",
                "C",
                "D",
                "E",
                "G",
            },
            // rids_in
            {
                "A",
                "C",
                "D",
                "F",
                "E",
            },
            // rids_out
            {
                "A",
                "C",
                "H",
                "F",
                "E",
            },
            // expected_reordered_chunk
            torch::tensor({
                {
                    {1, 1, 1, 1},    // A
                    {0, 0, 0, 0},    // B -> __blank
                    {2, 2, 2, 2},    // C
                    {3, 3, 3, 3},    // D
                    {5, 5, 5, 5},    // E
                    {0, 0, 0, 0},    // G -> __blank
                    {4, 4, 4, 4},    // F
                }
            }),
            // expected_out_ids
            {
                "A",
                "__blank",
                "C",
                "H",
                "E",
                "__blank",
                "F"
            },
        },

        TestCase{
            "Reordering. Five rows (dim = 1), input tensor is [1 x 5 x 4]. Unmatched in rows can be matched with empty previous out rows",
            // chunk
            torch::tensor({
                {
                    {1, 1, 1, 1},    // A
                    {2, 2, 2, 2},    // C
                    {3, 3, 3, 3},    // D -> H
                    {4, 4, 4, 4},    // F
                    {5, 5, 5, 5},    // E
                }
            }),
            // prev_rids_out
            {
                "A",
                "__blank",
                "C",
                "D",
                "E",
                "G",
            },
            // rids_in
            {
                "A",
                "C",
                "D",
                "F",
                "E",
            },
            // rids_out
            {
                "A",
                "C",
                "H",
                "F",
                "E",
            },
            // expected_reordered_chunk
            torch::tensor({
                {
                    {1, 1, 1, 1},    // A
                    {4, 4, 4, 4},    // __blank -> F
                    {2, 2, 2, 2},    // C
                    {3, 3, 3, 3},    // D
                    {5, 5, 5, 5},    // E
                    {0, 0, 0, 0},    // G -> __blank
                }
            }),
            // expected_out_ids
            {
                "A",
                "F",
                "C",
                "H",
                "E",
                "__blank",
            },
        },

        TestCase{
            "Reordering. Five rows (dim = 1), input tensor is [1 x 5 x 4]. Empty in rows can be matched with unmatched out rows",
            // chunk
            torch::tensor({
                {
                    {1, 1, 1, 1},    // A
                    {2, 2, 2, 2},    // C
                    {3, 3, 3, 3},    // D/H
                    {0, 0, 0, 0},    // __blank
                    {5, 5, 5, 5},    // E
                }
            }),
            // prev_rids_out
            {
                "A",
                "B",
                "C",
                "D",
                "E",
                "G",
            },
            // rids_in
            {
                "A",
                "C",
                "D",
                "__blank",
                "E",
            },
            // rids_out
            {
                "A",
                "C",
                "H",
                "__blank",
                "E",
            },
            // expected_reordered_chunk
            torch::tensor({
                {
                    {1, 1, 1, 1},    // A
                    {0, 0, 0, 0},    // B -> __blank
                    {2, 2, 2, 2},    // C
                    {3, 3, 3, 3},    // D/H
                    {5, 5, 5, 5},    // E
                    {0, 0, 0, 0},    // G -> __blank
                }
            }),
            // expected_out_ids
            {
                "A",
                "__blank",
                "C",
                "H",
                "E",
                "__blank",
            },
        },

        TestCase{
            "Reordering. Five rows (dim = 1), input tensor is [1 x 5 x 4]. Blanks can be matched with blanks",
            // chunk
            torch::tensor({
                {
                    {1, 1, 1, 1},    // A
                    {2, 2, 2, 2},    // C
                    {3, 3, 3, 3},    // D/H
                    {0, 0, 0, 0},    // __blank
                    {5, 5, 5, 5},    // E
                }
            }),
            // prev_rids_out
            {
                "A",
                "__blank",
                "C",
                "D",
                "E",
                "__blank",
            },
            // rids_in
            {
                "A",
                "C",
                "D",
                "__blank",
                "E",
            },
            // rids_out
            {
                "A",
                "C",
                "H",
                "__blank",
                "E",
            },
            // expected_reordered_chunk
            torch::tensor({
                {
                    {1, 1, 1, 1},    // A
                    {0, 0, 0, 0},    // __blank -> __blank
                    {2, 2, 2, 2},    // C
                    {3, 3, 3, 3},    // D/H
                    {5, 5, 5, 5},    // E
                    {0, 0, 0, 0},    // __blank -> __blank
                }
            }),
            // expected_out_ids
            {
                "A",
                "__blank",
                "C",
                "H",
                "E",
                "__blank",
            },
        },

        TestCase{
            "Reordering. Blank IDs are not matched by name, even if prev_rids_out and rids_in share a blank ID.",
            // chunk
            torch::tensor({
                {
                    {1, 1, 1, 1},    // N
                    {0, 0, 0, 0},    // __blank
                }
            }),
            // prev_rids_out
            {
                "__blank",
                "B",
            },
            // rids_in
            {
                "N",
                "__blank",
            },
            // rids_out
            {
                "oN",
                "oBlank",
            },
            // expected_reordered_chunk: N takes the blank row, the in blank takes row B.
            torch::tensor({
                {
                    {1, 1, 1, 1},    // N
                    {0, 0, 0, 0},    // __blank
                }
            }),
            // expected_out_ids
            {
                "oN",
                "oBlank",
            },
        },

        TestCase{
            "Reordering. Pairing does not depend on the order of the blank rows.",
            // chunk
            torch::tensor({
                {
                    {0, 0, 0, 0},    // __blank
                    {1, 1, 1, 1},    // A
                    {2, 2, 2, 2},    // N
                }
            }),
            // prev_rids_out
            {
                "A",
                "__blank",
                "B",
            },
            // rids_in
            {
                "__blank",
                "A",
                "N",
            },
            // rids_out
            {
                "oBlank",
                "oA",
                "oN",
            },
            // expected_reordered_chunk: N takes the out blank row, the in blank takes row B.
            torch::tensor({
                {
                    {1, 1, 1, 1},    // A
                    {2, 2, 2, 2},    // N
                    {0, 0, 0, 0},    // __blank
                }
            }),
            // expected_out_ids
            {
                "oA",
                "oN",
                "oBlank",
            },
        },

        TestCase{
            "No reordering. Valid order is kept when IDs are paired by index with at least one blank.",
            // chunk
            torch::tensor({
                {
                    {1, 1, 1, 1},    // __blank
                    {2, 2, 2, 2},    // N
                }
            }),
            // prev_rids_out
            {
                "__blank",
                "__blank",
            },
            // rids_in
            {
                "__blank",
                "N",
            },
            // rids_out
            {
                "oBlank",
                "oN",
            },
            // expected_reordered_chunk: the minimal pairing would swap the rows, but the order is valid.
            torch::tensor({
                {
                    {1, 1, 1, 1},    // __blank
                    {2, 2, 2, 2},    // N
                }
            }),
            // expected_out_ids
            {
                "oBlank",
                "oN",
            },
        },

        TestCase{
            "No reordering. Valid order is kept and the chunk is padded when prev_rids_out is longer.",
            // chunk
            torch::tensor({
                {
                    {1, 1, 1, 1},    // N
                    {2, 2, 2, 2},    // __blank
                }
            }),
            // prev_rids_out
            {
                "__blank",
                "__blank",
                "C",
            },
            // rids_in
            {
                "N",
                "__blank",
            },
            // rids_out
            {
                "oN",
                "oBlank",
            },
            // expected_reordered_chunk
            torch::tensor({
                {
                    {1, 1, 1, 1},    // N
                    {2, 2, 2, 2},    // __blank
                    {0, 0, 0, 0},    // __blank
                }
            }),
            // expected_out_ids
            {
                "oN",
                "oBlank",
                "__blank",
            },
        },

        TestCase{
            "Reordering. A common ID at a different index requires reordering even if all rows pair by index.",
            // chunk
            torch::tensor({
                {
                    {1, 1, 1, 1},    // __blank
                    {2, 2, 2, 2},    // A
                }
            }),
            // prev_rids_out
            {
                "A",
                "__blank",
            },
            // rids_in
            {
                "__blank",
                "A",
            },
            // rids_out
            {
                "oBlank",
                "oA",
            },
            // expected_reordered_chunk
            torch::tensor({
                {
                    {2, 2, 2, 2},    // A
                    {1, 1, 1, 1},    // __blank
                }
            }),
            // expected_out_ids
            {
                "oA",
                "oBlank",
            },
        },

        TestCase{
            "Reordering. Different named IDs at the same index require reordering and are not paired.",
            // chunk
            torch::tensor({
                {
                    {1, 1, 1, 1},    // C
                    {2, 2, 2, 2},    // D
                }
            }),
            // prev_rids_out
            {
                "A",
                "B",
            },
            // rids_in
            {
                "C",
                "D",
            },
            // rids_out
            {
                "oC",
                "oD",
            },
            // expected_reordered_chunk
            torch::tensor({
                {
                    {0, 0, 0, 0},    // __blank
                    {0, 0, 0, 0},    // __blank
                    {1, 1, 1, 1},    // C
                    {2, 2, 2, 2},    // D
                }
            }),
            // expected_out_ids
            {
                "__blank",
                "__blank",
                "oC",
                "oD",
            },
        },

        TestCase{
            "Reordering. Duplicate read IDs cause the final chunk to have more rows than before.",
            // chunk
            torch::tensor({
                {
                    {1, 1, 1, 1},    // A
                    {2, 2, 2, 2},    // C
                    {3, 3, 3, 3},    // D -> H
                    {4, 4, 4, 4},    // E
                    {5, 5, 5, 5},    // X
                    {6, 6, 6, 6},    // X
                }
            }),
            // prev_rids_out
            {
                "A",
                "C",
                "C",
                "C",
                "D",
                "E",
            },
            // rids_in
            {
                "A",
                "C",
                "D",
                "E",
                "X",
                "X",
            },
            // rids_out
            {
                "A",
                "C",
                "H",
                "E",
                "X",
                "X",
            },
            // expected_reordered_chunk
            torch::tensor({
                {
                    {1, 1, 1, 1},    // A
                    {2, 2, 2, 2},    // C
                    {2, 2, 2, 2},    // C
                    {2, 2, 2, 2},    // C
                    {3, 3, 3, 3},    // H
                    {4, 4, 4, 4},    // E
                    {5, 5, 5, 5},    // X
                    {6, 6, 6, 6},    // X
                }
            }),
            // expected_out_ids
            {
                "A",
                "C",
                "C",
                "C",
                "H",
                "E",
                "X",
                "X",
            },
        },
    }));
    // clang-format on

    CATCH_INFO(TEST_GROUP << " Test name: " << test_case.test_name);

    const auto [result_reordered_chunk, result_out_ids] = reorder_chunk(
            test_case.chunk, test_case.prev_rids_out, test_case.rids_in, test_case.rids_out);

    if (!test_case.chunk.defined()) {
        CATCH_CHECK(!test_case.expected_reordered_chunk.defined());
        CATCH_CHECK(std::empty(test_case.expected_out_ids));
    } else {
        CATCH_CHECK(test_case.expected_reordered_chunk.equal(result_reordered_chunk));
        CATCH_CHECK(test_case.expected_out_ids == result_out_ids);
    }
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
