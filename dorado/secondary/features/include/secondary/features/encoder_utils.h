#pragma once

#include <ATen/ATen.h>

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace dorado::secondary {

/**
 * \brief Describes how the rows (reads) of a chunk are placed so that they line up with the
 *          rows of the previous chunk.
 *        src_rows vector stores for each row in the reordered output the row index in the
 *          input chunk, or -1 for an empty (zero) row.
 *        next_rids_out stores the right-read_ids of the reordered output. Empty rows are
 *          labelled with BLANK_READ_PLACEHOLDER.
 *        requires_reorder is true if any reordering is needed, or false if all input rows keep
 *          their position (output rows >= the input depth are empty).
 *        The sizes of src_rows and next_rids_out are the depth of the output chunk.
 */
struct ChunkReorderPlan {
    std::vector<int64_t> src_rows;
    std::vector<std::string> next_rids_out;
    bool requires_reorder = true;
};

/**
 * \brief Plans the reordering of a chunk. Rows are matched by read ID, blank IDs can pair with any
 *          unmatched row, and two different named reads are never placed in the same row.
 * \param prev_rids_out Right read IDs of the previous (reordered) chunk.
 * \param rids_in Left read IDs of this chunk, one per row.
 * \param rids_out Right read IDs of this chunk, one per row.
 */
ChunkReorderPlan plan_chunk_reorder(const std::vector<std::string>& prev_rids_out,
                                    const std::vector<std::string>& rids_in,
                                    const std::vector<std::string>& rids_out);

/**
 * \brief Concatenates neighboring chunks along dim 0, lining up their rows (dim 1) by read ID.
 *          All chunks are planned from the read IDs first, so that the output is allocated once at its
 *          final size and every chunk is copied once. Padding is a single zero fill of the output.
 * \param chunks Tensors of shape [positions_i, reads_i, ...]. Consumed: each is released once copied.
 * \param read_ids_left Left read IDs of each chunk, sizes must match reads_i.
 * \param read_ids_right Right read IDs of each chunk, sizes must match reads_i.
 * \returns The merged tensor, the left and right read IDs of the merged chunks (padded with blanks to
              the new depth).
 * \throws std::runtime_error if any chunk has invalid shape or if chunks are incompatible for merging.
 */
std::tuple<at::Tensor, std::vector<std::string>, std::vector<std::string>> merge_read_chunks(
        std::vector<at::Tensor>& chunks,
        const std::vector<std::vector<std::string>>& read_ids_left,
        const std::vector<std::vector<std::string>>& read_ids_right);

/**
 * \brief In the read-level feature tensor, this function finds and removes
 *          minor columns which have all bases equal to zero (like padding).
 *          It also relabels the `positions_minor` and returns them as well
 *          to avoid edge cases with right-aligning bases in insertion regions.
 * \param feature_tensor Input tensor of features of shape [positions x reads x features].
 *                          The `features` column consists of one or more values (typically 5-7),
 *                          where the first element is the base of that read at that position.
 *                          Bases are non-zero, so a value of 0 means padding.
 * \param positions_major Vector of major positions, the same width as the number of rows in
 *                          the feature_tensor.
 * \param positions_minor Vector of minor positions, the same width as the number of rows in
 *                          the feature_tensor.
 * \returns Three values <filtered_tensor, new_positions_major, new_positions_minor>.
 *          The `filtered_tensor` does not contain any empty minor columns.
 *          The `new_positions_minor` is relabeled to avoid discontinuities in the minor
 *          position IDs.
 *          The `new_positions_minor` is reduced to match the tensor.
 */
std::tuple<at::Tensor, std::vector<int64_t>, std::vector<int64_t>> filter_empty_minor_columns(
        const at::Tensor& feature_tensor,
        const std::vector<int64_t>& positions_major,
        const std::vector<int64_t>& positions_minor);

/**
 * \brief This function returns a tensor with an integer encoding of a sequence scattered over
 *        major positions.
 * \param positions_major Vector of major positions.
 * \param positions_minor Vector of minor positions.
 * \param draft_seq Draft/reference sequence.
 * \returns A tensor with integer encoding of the subsequence (including gaps) spanned by the
 *          positions vectors. The length of this tensor is the same as the positions vectors.
 */
at::Tensor draft_encoding_from_seq(const std::vector<int64_t>& positions_major,
                                   const std::vector<int64_t>& positions_minor,
                                   const std::string_view draft_seq);

/**
 * \brief This function returns a vector with an integer encoding of a sequence.
 * \param sequence Draft/reference sequence.
 * \returns A vector with gap-compressed integer encoding of the sequence. Any
 *          unrecognised characters are ignored.
 */
std::vector<int8_t> encode_sequence(const std::string_view sequence);

}  // namespace dorado::secondary
