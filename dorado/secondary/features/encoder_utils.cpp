#include "encoder_utils.h"

#include "torch_utils/tensor_utils.h"
#include "utils/container_utils.h"

#include <spdlog/spdlog.h>

#include <cstddef>
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

}  // namespace dorado::secondary
