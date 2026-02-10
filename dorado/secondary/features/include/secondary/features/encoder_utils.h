#pragma once

#include <ATen/ATen.h>

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace dorado::secondary {

std::tuple<at::Tensor, std::vector<std::string>> reorder_chunk(
        const at::Tensor& chunk,
        const std::vector<std::string>& prev_rids_out,
        const std::vector<std::string>& rids_in,
        const std::vector<std::string>& rids_out);

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

}  // namespace dorado::secondary
