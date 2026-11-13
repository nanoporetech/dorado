#include "secondary/features/encoder_utils.h"

#include "secondary/features/medaka_read_matrix_constants.h"
#include "torch_utils/tensor_utils.h"
#include "utils/container_utils.h"
#include "utils/log_utils.h"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace dorado::secondary {

namespace {

// Rows without a read are labelled with blank placeholder ID. Blank IDs
// are never matched by name, but they can be paired with any unmatched row.
bool is_blank(const std::string_view rid) { return rid == BLANK_READ_PLACEHOLDER; }

/**
 * \brief Checks whether the rows of the new chunk can stay where they are. This is the case if
 *          (1) any read ID common to both vectors is at the same index in both, and
 *          (2) no two different named (non-blank) IDs would share a row.
 *          Rows paired by index where at least one ID is blank are acceptable.
 */
bool ordering_is_valid(const std::vector<std::string>& prev_rids_out,
                       const std::vector<std::string>& rids_in) {
    const size_t num_shared = std::min(std::size(prev_rids_out), std::size(rids_in));
    for (size_t i = 0; i < num_shared; ++i) {
        if ((prev_rids_out[i] != rids_in[i]) && !is_blank(prev_rids_out[i]) &&
            !is_blank(rids_in[i])) {
            return false;
        }
    }

    std::unordered_map<std::string_view, size_t> prev_rids_out_map;
    for (size_t i = 0; i < std::size(prev_rids_out); ++i) {
        if (!is_blank(prev_rids_out[i])) {
            prev_rids_out_map.try_emplace(prev_rids_out[i], i);
        }
    }
    for (size_t i = 0; i < std::size(rids_in); ++i) {
        const auto it = is_blank(rids_in[i]) ? std::end(prev_rids_out_map)
                                             : prev_rids_out_map.find(rids_in[i]);
        if ((it != std::end(prev_rids_out_map)) && (it->second != i)) {
            return false;
        }
    }

    return true;
}

}  // namespace

ChunkReorderPlan plan_chunk_reorder(const std::vector<std::string>& prev_rids_out,
                                    const std::vector<std::string>& rids_in,
                                    const std::vector<std::string>& rids_out) {
    // The rows of the chunk keep their place, and rows are added at the end if the previous chunk is deeper.
    const auto make_identity_plan = [&]() {
        const int64_t num_in = std::ssize(rids_in);
        const int64_t num_out = std::max(std::size(prev_rids_out), std::size(rids_in));
        ChunkReorderPlan plan{
                .src_rows = std::vector<int64_t>(num_out, -1),
                .next_rids_out = rids_out,
                .requires_reorder = false,
        };
        std::iota(std::begin(plan.src_rows), std::begin(plan.src_rows) + num_in, 0);
        plan.next_rids_out.resize(num_out, std::string(BLANK_READ_PLACEHOLDER));
        return plan;
    };

    if (prev_rids_out == rids_in) {
        DORADO_LOG(trace, "[plan_chunk_reorder] Read ids match, skipping reordering.");
        return make_identity_plan();
    }

    DORADO_LOG(trace, "[plan_chunk_reorder] prev_rids_out.size() = {}, rids_in.size() = {}",
               std::size(prev_rids_out), std::size(rids_in));

    // If the rows are already in a valid order, avoid reordering. This can leave unnecessary
    // empty rows, but is cheaper than permuting the chunk.
    if (ordering_is_valid(prev_rids_out, rids_in)) {
        DORADO_LOG(trace, "[plan_chunk_reorder] Existing row order is valid, skipping reordering.");
        return make_identity_plan();
    }

    // Create a lookup of the named reads.
    std::unordered_map<std::string_view, int64_t> rids_in_map;
    for (int64_t i = 0; i < std::ssize(rids_in); ++i) {
        if (!is_blank(rids_in[i])) {
            rids_in_map[rids_in[i]] = i;
        }
    }

    // Find the indices of the out reads in the in reads.
    std::vector<int64_t> new_indices(std::size(prev_rids_out), -1);
    for (int64_t i = 0; i < std::ssize(prev_rids_out); ++i) {
        const std::string_view rid = prev_rids_out[i];
        const auto it = rids_in_map.find(rid);
        new_indices[i] = ((it == std::end(rids_in_map)) || is_blank(rid)) ? -1 : it->second;
    }

    // Find missing out indices.
    std::vector<int64_t> named_out;
    std::vector<int64_t> blank_out;
    for (int64_t i = 0; i < std::ssize(new_indices); ++i) {
        if (new_indices[i] == -1) {
            if (is_blank(prev_rids_out[i])) {
                blank_out.emplace_back(i);
            } else {
                named_out.emplace_back(i);
            }
        }
    }

    // Find missing in indices.
    const std::unordered_set<int64_t> new_indices_set(std::begin(new_indices),
                                                      std::end(new_indices));
    std::vector<int64_t> named_in;
    std::vector<int64_t> blank_in;
    for (int64_t i = 0; i < std::ssize(rids_in); ++i) {
        if (!new_indices_set.contains(i)) {
            if (is_blank(rids_in[i])) {
                blank_in.emplace_back(i);
            } else {
                named_in.emplace_back(i);
            }
        }
    }

    DORADO_LOG(trace,
               "[plan_chunk_reorder] named_in.size() = {}, blank_in.size() = {}, "
               "named_out.size() = {}, blank_out.size() = {}",
               std::size(named_in), std::size(blank_in), std::size(named_out),
               std::size(blank_out));

    // Fill out the gaps in the array with some of the extra indices.
    // A pair of reads can be joined if either side is blank. We disallow joining of two named
    // reads as this would combine multiple reads into contiguous read data in the tensor.
    // Named reads take blank rows first, then blanks take the remaining named rows, which
    // minimises the number of rows regardless of the row order.
    const auto pair_up = [&new_indices](std::vector<int64_t>& in_indices,
                                        std::vector<int64_t>& out_indices) {
        const size_t num_pairs = std::min(std::size(in_indices), std::size(out_indices));
        for (size_t i = 0; i < num_pairs; ++i) {
            new_indices[out_indices[i]] = in_indices[i];
        }
        in_indices.erase(std::begin(in_indices), std::begin(in_indices) + num_pairs);
        out_indices.erase(std::begin(out_indices), std::begin(out_indices) + num_pairs);
    };
    pair_up(named_in, blank_out);
    pair_up(blank_in, named_out);
    pair_up(blank_in, blank_out);

    // Add remaining missing in-indices.
    named_in.insert(std::end(named_in), std::cbegin(blank_in), std::cend(blank_in));
    std::sort(std::begin(named_in), std::end(named_in));
    new_indices.insert(std::end(new_indices), std::cbegin(named_in), std::cend(named_in));

    // Update read_ids_out for the next chunk.
    ChunkReorderPlan plan{
            .src_rows = std::move(new_indices),
            .next_rids_out = {},
            .requires_reorder = true,
    };
    plan.next_rids_out.reserve(std::size(plan.src_rows));
    for (const int64_t idx : plan.src_rows) {
        plan.next_rids_out.emplace_back(idx == -1 ? std::string(BLANK_READ_PLACEHOLDER)
                                                  : rids_out[idx]);
    }

    return plan;
}

namespace {

/**
 * \brief Copies a chunk of shape [rows, depth_in, ...] into a block of shape [rows, depth_out, ...]
 *          following the plan. Rows which are not assigned a source row are not written.
 *          Runs of consecutive source rows are copied together, which needs no intermediate
 *          tensor and only a single copy if the rows keep their order.
 */
void place_chunk(const at::Tensor& block, const at::Tensor& chunk, const ChunkReorderPlan& plan) {
    const int64_t chunk_rows = chunk.size(1);
    if (!plan.requires_reorder) {
        block.narrow(1, 0, chunk_rows).copy_(chunk);
        return;
    }
    const int64_t out_rows = std::ssize(plan.src_rows);
    int64_t dst = 0;
    while (dst < out_rows) {
        const int64_t src = plan.src_rows[dst];
        if (src == -1) {
            ++dst;
            continue;
        }
        int64_t run_length = 1;
        while (((dst + run_length) < out_rows) &&
               (plan.src_rows[dst + run_length] == (src + run_length))) {
            ++run_length;
        }
        block.narrow(1, dst, run_length).copy_(chunk.narrow(1, src, run_length));
        dst += run_length;
    }
}

}  // namespace

std::tuple<at::Tensor, std::vector<std::string>> reorder_chunk(
        const at::Tensor& chunk,
        const std::vector<std::string>& prev_rids_out,
        const std::vector<std::string>& rids_in,
        const std::vector<std::string>& rids_out) {
    if (!chunk.defined()) {
        return {};
    }

    ChunkReorderPlan plan = plan_chunk_reorder(prev_rids_out, rids_in, rids_out);

    const int64_t out_depth = std::ssize(plan.src_rows);
    if (!plan.requires_reorder && (chunk.size(1) == out_depth)) {
        return {chunk, std::move(plan.next_rids_out)};
    }

    std::vector<int64_t> shape = chunk.sizes().vec();
    shape[1] = out_depth;
    at::Tensor reordered_chunk = at::zeros(shape, chunk.options());
    place_chunk(reordered_chunk, chunk, plan);

    return {reordered_chunk, std::move(plan.next_rids_out)};
}

std::tuple<at::Tensor, std::vector<std::string>, std::vector<std::string>> merge_read_chunks(
        std::vector<at::Tensor>& chunks,
        const std::vector<std::vector<std::string>>& read_ids_left,
        const std::vector<std::vector<std::string>>& read_ids_right) {
    if (std::empty(chunks)) {
        return {};
    }
    if ((std::size(read_ids_left) != std::size(chunks)) ||
        (std::size(read_ids_right) != std::size(chunks))) {
        throw std::runtime_error{
                "Cannot merge read chunks, mismatch between the number of read ID vectors ("
                "left = " +
                std::to_string(std::size(read_ids_left)) +
                ", right = " + std::to_string(std::size(read_ids_right)) +
                ") and the number of chunks = " + std::to_string(std::size(chunks)) + "."};
    }
    if (std::size(chunks) == 1) {
        return std::make_tuple(std::move(chunks.front()), read_ids_left.front(),
                               read_ids_right.front());
    }

    // Plan all of the chunks using only the read IDs, to find the final size before allocating.
    std::vector<ChunkReorderPlan> plans;
    plans.reserve(std::size(chunks));
    int64_t num_positions = 0;
    int64_t max_depth = 0;

    // The first chunk defines the expected shape, dtype and device, so it is checked up front.
    if (!chunks.front().defined()) {
        throw std::runtime_error{"Cannot merge read chunks, chunk 0 is undefined."};
    }
    if (chunks.front().dim() < 2) {
        throw std::runtime_error{"Cannot merge read chunks with dimension < 2"};
    }
    std::vector<int64_t> shape = chunks.front().sizes().vec();
    const std::vector<int64_t> inner_shape{std::begin(shape) + 2, std::end(shape)};
    const at::ScalarType dtype = chunks.front().scalar_type();
    const at::Device device = chunks.front().device();
    const auto check_compatible_tensor = [&inner_shape, &dtype, &device](const at::Tensor& chunk) {
        return (chunk.sizes().slice(2).vec() == inner_shape) && (chunk.scalar_type() == dtype) &&
               (chunk.device() == device);
    };

    for (size_t i = 0; i < std::size(chunks); ++i) {
        const at::Tensor& chunk = chunks[i];
        if (!chunk.defined()) {
            throw std::runtime_error{"Cannot merge read chunks, chunk " + std::to_string(i) +
                                     " is undefined."};
        }
        if ((chunk.dim() < 2) || (chunk.size(1) != std::ssize(read_ids_left[i]))) {
            throw std::runtime_error{"Cannot merge read chunks, chunk " + std::to_string(i) +
                                     " has an unexpected shape = [" +
                                     utils::tensor_shape_as_string(chunk) + "], expected [" +
                                     std::to_string(std::size(read_ids_left[i])) + "]."};
        }
        if (!check_compatible_tensor(chunk)) {
            throw std::runtime_error{
                    "Cannot merge incompatible read chunks, expected "
                    "(inner_shape = " +
                    utils::print_container_as_string(inner_shape, ",", true) +
                    ", dtype = " + at::toString(dtype) + ", device = " + at::toString(device) +
                    "); chunk " + std::to_string(i) + "(inner_shape = " +
                    utils::print_container_as_string(chunk.sizes().slice(2).vec(), ",", true) +
                    ", dtype = " + at::toString(chunk.scalar_type()) +
                    ", device = " + at::toString(chunk.device()) + ")."};
        }
        num_positions += chunk.size(0);

        // The first chunk is planned against its own left IDs, which gives the identity plan.
        const auto& prev_rids_out = (i == 0) ? read_ids_left[0] : plans.back().next_rids_out;
        plans.emplace_back(plan_chunk_reorder(prev_rids_out, read_ids_left[i], read_ids_right[i]));
        max_depth = std::max<int64_t>(max_depth, std::ssize(plans.back().src_rows));
    }

    // Allocate the output once. Zero filling it is the only padding done.
    shape[0] = num_positions;
    shape[1] = max_depth;
    DORADO_LOG(trace, "[merge_read_chunks] Allocating tensor of shape {}",
               utils::print_container_as_string(shape, ",", true));
    at::Tensor merged = at::zeros(shape, chunks.front().options());

    int64_t offset = 0;
    for (size_t i = 0; i < std::size(chunks); ++i) {
        const int64_t chunk_positions = chunks[i].size(0);
        place_chunk(merged.narrow(0, offset, chunk_positions), chunks[i], plans[i]);
        offset += chunk_positions;
        // Release the input as soon as it has been copied.
        chunks[i] = at::Tensor{};
    }

    // Allocate and pad the read_id vectors as needed
    std::vector<std::string> merged_read_ids_left = read_ids_left.front();
    std::vector<std::string> merged_read_ids_right = std::move(plans.back().next_rids_out);
    merged_read_ids_left.resize(max_depth, std::string(BLANK_READ_PLACEHOLDER));
    merged_read_ids_right.resize(max_depth, std::string(BLANK_READ_PLACEHOLDER));

    return std::make_tuple(std::move(merged), std::move(merged_read_ids_left),
                           std::move(merged_read_ids_right));
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

    using namespace at::indexing;

    // Extract base feature channel: [positions, reads]
    const at::Tensor bases = feature_tensor.index({Ellipsis, 0});

    // Convert positions_minor to a Tensor and move to the same device as the feature_tensor.
    const at::Tensor pos_minor_t =
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast): from_blob() views non-const data but we clone() it.
            at::from_blob(const_cast<int64_t*>(positions_minor.data()),
                          {std::ssize(positions_minor)}, at::TensorOptions().dtype(at::kLong))
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

namespace {
// Converts ASCII to BAM nibble encoding.
// Copied from htslib/hts.c since linking to <htslib/hts.h> fails on windows.
static constexpr unsigned char SEQ_NT16_TABLE[256]{
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 1,  2,  4,  8,  15, 15, 15, 15, 15, 15, 15, 15, 15, 0,  15, 15, 15, 1,
        14, 2,  13, 15, 15, 4,  11, 15, 15, 12, 15, 3,  15, 15, 15, 15, 5,  6,  8,  8,  7,  9,
        15, 10, 15, 15, 15, 15, 15, 15, 15, 1,  14, 2,  13, 15, 15, 4,  11, 15, 15, 12, 15, 3,
        15, 15, 15, 15, 5,  6,  8,  8,  7,  9,  15, 10, 15, 15, 15, 15, 15, 15,

        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15};

}  // namespace

at::Tensor draft_encoding_from_seq(const std::vector<int64_t>& positions_major,
                                   const std::vector<int64_t>& positions_minor,
                                   const std::string_view draft_seq) {
    const auto opts = at::TensorOptions().dtype(at::kInt);
    const int64_t sample_size = std::ssize(positions_major);
    if (sample_size == 0) {
        return at::empty({});
    }
    int64_t pos = positions_major[0];
    // if first position is minor, start at the next reference base
    if (positions_minor[0] != 0) {
        ++pos;
    }
    const int64_t seq_end = positions_major[sample_size - 1] + 1;  // end-exclusive

    at::Tensor draft_seq_tensor = at::full({sample_size}, DEL_VAL, opts);

    if (seq_end > std::ssize(draft_seq)) {
        throw std::runtime_error{"Reference coordinates " + std::to_string(pos) + "-" +
                                 std::to_string(seq_end) +
                                 " extend beyond the provided reference length (" +
                                 std::to_string(draft_seq.length()) + ")."};
    }
    if (pos == seq_end) {
        // Entire chunk is in an insertion, nothing to do
        return draft_seq_tensor;
    }
    for (int64_t idx = 0; idx < sample_size; ++idx) {
        if (positions_minor[idx] == 0) {
            const auto base_encoding =
                    NUM_TO_COUNT_BASE_SYMM[SEQ_NT16_TABLE[static_cast<uint8_t>(draft_seq[pos])]];
            if (base_encoding >= 1) {
                draft_seq_tensor.index({idx}) = base_encoding;
            }
            ++pos;
        }
    }
    return draft_seq_tensor;
}

std::vector<int8_t> encode_sequence(const std::string_view sequence) {
    std::vector<int8_t> encoded_sequence;
    encoded_sequence.reserve(std::size(sequence));
    for (const char c : sequence) {
        int8_t base_encoding = NUM_TO_COUNT_BASE_SYMM[SEQ_NT16_TABLE[static_cast<uint8_t>(c)]];
        if ((base_encoding >= 1) && (base_encoding != DEL_VAL)) {
            encoded_sequence.emplace_back(base_encoding);
        }
    }
    return encoded_sequence;
}

}  // namespace dorado::secondary
