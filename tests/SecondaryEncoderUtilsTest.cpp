#include "secondary/features/encoder_utils.h"

#include <ATen/ATen.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <vector>

#define TEST_GROUP "[SecondaryEncoderUtils]"

namespace dorado::secondary::tests {
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
        ///     4. Fill empty new_indices with missing_in_indices:
        ///             new_indices = {0, 3, 1, 2, 4, -1}
        ///     5. If len(missing_in_indices) > len(missing_out_indices), append remaining to new_indices.
        ///             -> Not the case here
        ///     6. Reorder chunk: permute with new_indices as the key, if new_indices[i] != -1.
        ///     7. Create thew next_rids_out. For new_indices[i] which exists, keep existing rids_out[i]. Otherwise,
        ///         set new_indices[i] = "__inserted_{i}".
        ///
        /// So, the next_rids_out is filled like this:
        ///         new_indices = {0, -1, 1, 2, 4, -1}
        ///                     V
        ///         new_indices = {0, 3, 1, 2, 4, -1}

        ///         expected_out_ids = {A, F, C, H, E, __insert   }
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
                    {4, 4, 4, 4},    // F
                    {2, 2, 2, 2},    // C
                    {3, 3, 3, 3},    // D -> H
                    {5, 5, 5, 5},    // E
                    {0, 0, 0, 0},    // __inserted_5
                }
            }),
            // expected_out_ids
            {
                "A",
                "F",
                "C",
                "H",
                "E",
                "__inserted_5",
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
