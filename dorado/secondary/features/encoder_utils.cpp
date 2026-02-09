#include "secondary/features/encoder_utils.h"

#include "torch_utils/tensor_utils.h"
#include "utils/container_utils.h"

#include <spdlog/spdlog.h>
#include <torch/types.h>

#include <cstddef>
#include <span>
#include <unordered_map>
#include <unordered_set>

#ifdef NDEBUG
#define LOG_TRACE(...)
#else
#define LOG_TRACE(...) spdlog::trace(__VA_ARGS__)
#endif

namespace dorado::secondary {

std::tuple<at::Tensor, std::vector<std::string>> reorder_chunk(
        const at::Tensor& chunk,
        const std::vector<std::string>& prev_rids_out,
        const std::vector<std::string>& rids_in,
        const std::vector<std::string>& rids_out) {
    if (!chunk.defined()) {
        return {};
    }

    LOG_TRACE(
            "[reorder_chunk] prev_rids_out.size() = {}, rids_in.size() = {}, "
            "chunk.shape = [{}]",
            std::size(prev_rids_out), std::size(rids_in), utils::tensor_shape_as_string(chunk));

    // Create a lookup.
    std::unordered_map<std::string_view, int64_t> rids_in_map;
    for (int64_t i = 0; i < std::ssize(rids_in); ++i) {
        rids_in_map[rids_in[i]] = i;
    }

    // Find the indices of the out reads in the in reads.
    std::vector<int64_t> new_indices(std::size(prev_rids_out), -1);
    for (int64_t i = 0; i < std::ssize(prev_rids_out); ++i) {
        const std::string_view rid = prev_rids_out[i];
        const auto it = rids_in_map.find(rid);
        new_indices[i] = (it != std::end(rids_in_map)) ? it->second : -1;
    }

    // Find missing out indices.
    std::vector<int64_t> missing_out_indices;
    for (int64_t i = 0; i < std::ssize(new_indices); ++i) {
        if (new_indices[i] == -1) {
            missing_out_indices.emplace_back(i);
        }
    }

    // Find missing in indices.
    const std::unordered_set<int64_t> new_indices_set(std::begin(new_indices),
                                                      std::end(new_indices));
    std::vector<int64_t> missing_in_indices;
    for (int64_t i = 0; i < std::ssize(rids_in); ++i) {
        if (!new_indices_set.contains(i)) {
            missing_in_indices.emplace_back(i);
        }
    }

    LOG_TRACE(
            "[reorder_chunk] missing_in_indices.size() = {}, "
            "missing_out_indices.size() = {}",
            std::size(missing_in_indices), std::size(missing_out_indices));
    LOG_TRACE("[reorder_chunk] missing_in_indices: {}",
              utils::print_container_as_string(missing_in_indices, ", ", true));
    LOG_TRACE("[reorder_chunk] missing_out_indices: {}",
              utils::print_container_as_string(missing_out_indices, ", ", true));

    // Fill out the gaps in the array with some of the extra indices.
    for (size_t i = 0; i < std::min(std::size(missing_out_indices), std::size(missing_in_indices));
         ++i) {
        new_indices[missing_out_indices[i]] = missing_in_indices[i];
    }

    // Add remaining missing in-indices.
    if (std::size(missing_in_indices) > std::size(missing_out_indices)) {
        new_indices.insert(std::end(new_indices),
                           std::cbegin(missing_in_indices) + std::size(missing_out_indices),
                           std::cend(missing_in_indices));
    }

    // Permute.
    at::Tensor reordered_chunk = at::zeros(
            {chunk.size(0),
             std::max({std::ssize(new_indices), std::ssize(rids_in), std::ssize(prev_rids_out)}),
             chunk.size(2)},
            chunk.options());
    for (int64_t i = 0; i < std::ssize(new_indices); ++i) {
        if (new_indices[i] == -1) {
            continue;
        }
        reordered_chunk.index_put_(
                {at::indexing::Slice(), i, at::indexing::Slice()},
                chunk.index({at::indexing::Slice(), new_indices[i], at::indexing::Slice()}));
    }

    // Update read_ids_out for the next chunk.
    std::vector<std::string> next_rids_out(std::size(new_indices));
    for (int64_t i = 0; i < std::ssize(new_indices); ++i) {
        const int64_t idx = new_indices[i];
        next_rids_out[i] = (idx == -1) ? ("__inserted_" + std::to_string(i)) : rids_out[idx];
    }

    return std::tuple(std::move(reordered_chunk), std::move(next_rids_out));
}

namespace {
std::pair<std::vector<int64_t>, std::vector<int64_t>> relabel_positions(
        const std::span<const int64_t> positions_major,
        const std::span<const int64_t> positions_minor,
        const std::span<const bool> keep_mask) {
    // Sanity check.
    if ((std::size(positions_major) != std::size(keep_mask)) ||
        (std::size(positions_minor) != std::size(keep_mask))) {
        throw std::runtime_error{
                "Cannot relabel positions because the positions_major/positions_minor/keep_mask "
                "are of "
                "different sizes. positions_major.size = " +
                std::to_string(std::size(positions_major)) +
                ", positions_minor.size = " + std::to_string(std::size(positions_minor)) +
                ", keep_mask.size = " + std::to_string(std::size(keep_mask))};
    }

    const int64_t num_positions = std::ssize(positions_major);

    // Count the number of kept positions.
    int64_t num_keepers = 0;
    for (int64_t i = 0; i < num_positions; ++i) {
        num_keepers += keep_mask[i];
    }

    // Extract the keepers. The +1 is for book keeping to avoid a branch and to
    // remove trailing filtered elements.
    std::vector<int64_t> ret_major(num_keepers + 1, 0);
    std::vector<int64_t> ret_minor(num_keepers + 1, 0);
    for (int64_t i = 0, j = 0; i < num_positions; ++i) {
        ret_major[j] = positions_major[i];
        ret_minor[j] = positions_minor[i];
        j += keep_mask[i];
    }
    // Discard the last book keeping element.
    ret_major.resize(num_keepers);
    ret_minor.resize(num_keepers);

    // Sanity check for the keepers. If there is a switch between coordinates
    // in the ret_major array, but the corresponding ret_minor > 0, this means we
    // removed a major positions during filtering. This shouldn't happen.
    for (int64_t i = 1; i < num_keepers; ++i) {
        if ((ret_major[i] != ret_major[i - 1]) && (ret_minor[i] > 0)) {
            throw std::runtime_error{
                    "Error encountered while condendsing the feature tensor. A major position was "
                    "removed."};
        }
    }

    // If the window begins with a minor position and if that minor position gets removed, the
    // remaining minor positions need to start with the same minor value otherwise merging this window
    // later will not be successful.
    // But, if the major coordinate changes, this means we completely removed ALL minor columns on the left side,
    // so reset the minor counting.
    int64_t prev_minor = (positions_major[0] == ret_major[0]) ? positions_minor[0] : 0;
    for (int64_t i = 0; i < num_keepers; ++i) {
        if ((i > 0) && (ret_major[i] != ret_major[i - 1])) {
            prev_minor = 0;
        }
        ret_minor[i] = prev_minor;
        ++prev_minor;
    }

    return {ret_major, ret_minor};
}
}  // namespace

std::tuple<at::Tensor, std::vector<int64_t>, std::vector<int64_t>> filter_empty_minor_columns(
        const at::Tensor& feature_tensor,
        const std::vector<int64_t>& positions_major,
        const std::vector<int64_t>& positions_minor) {
    TORCH_CHECK(feature_tensor.defined(), "Uninitialized input tensor.");
    TORCH_CHECK(feature_tensor.dim() == 3, "Expected [positions, reads, features].");
    TORCH_CHECK(feature_tensor.size(2) > 0, "The features dim must be > 0.");
    TORCH_CHECK(feature_tensor.size(0) == std::ssize(positions_minor),
                "The positions_minor vector does not match in size to the input feature_tensor. "
                "positions_minor.size = ",
                std::size(positions_minor),
                ", feature_tensor.shape = ", utils::tensor_shape_as_string(feature_tensor));

    using namespace torch::indexing;

    // Extract base feature channel: [positions, reads]
    const at::Tensor bases = feature_tensor.index({Ellipsis, 0});

    // Convert positions_minor to a Tensor and move to the same device as the feature_tensor.
    const at::Tensor pos_minor_t = torch::from_blob(const_cast<int64_t*>(positions_minor.data()),
                                                    {std::ssize(positions_minor)},
                                                    torch::TensorOptions().dtype(torch::kInt64))
                                           .clone()
                                           .to(feature_tensor.device());

    const at::Tensor is_major = (pos_minor_t == 0);  // Shape: [positions]
    const at::Tensor pos_has_base = bases.any(1);    // Reduce dimension 1, shape: [positions]
    const at::Tensor pos_mask = is_major | pos_has_base;

    const at::Tensor pos_idx = at::nonzero(pos_mask).squeeze();

    // Select along dim 0, then dim 1
    at::Tensor ret_tensor =
            feature_tensor.index_select(/*dim=*/0, pos_idx);  // [positions_kept, reads, features]

    const std::span<const bool> keep_mask(pos_mask.data_ptr<bool>(), pos_mask.size(0));

    auto [ret_pos_major, ret_pos_minor] =
            relabel_positions(positions_major, positions_minor, keep_mask);

    return {std::move(ret_tensor), std::move(ret_pos_major), std::move(ret_pos_minor)};
}

}  // namespace dorado::secondary
