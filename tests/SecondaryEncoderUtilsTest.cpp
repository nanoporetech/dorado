#include "../dorado/secondary/features/encoder_utils.h"

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

}  // namespace dorado::secondary::tests
