#include "secondary/consensus/variant_calling_sample.h"

#include "secondary/consensus/consensus_utils.h"
#include "secondary/consensus/sample.h"
#include "secondary/consensus/sample_trimming.h"
#include "secondary/features/decoder_base.h"
#include "secondary/features/encoder_utils.h"
#include "secondary/features/medaka_read_matrix_constants.h"
#include "torch_utils/tensor_utils.h"
#include "utils/log_utils.h"

#include <algorithm>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace dorado::secondary {

void VariantCallingSample::validate() const {
    if (seq_id < 0) {
        std::ostringstream oss;
        oss << "VariantCallingSample::seq_id < 0. seq_id = " << seq_id;
        throw std::runtime_error(oss.str());
    }

    // Validate lengths.
    if (std::size(positions_major) != std::size(positions_minor)) {
        std::ostringstream oss;
        oss << "VariantCallingSample::positions_major and positions_minor are not of same size. "
               "positions_major.size = "
            << std::size(positions_major)
            << ", positions_minor.size = " << std::size(positions_minor);
        throw std::runtime_error(oss.str());
    }

    if (!logits.defined()) {
        throw std::runtime_error("VariantCallingSample::logits tensor is not defined!");
    }
    if (!depth.defined()) {
        throw std::runtime_error("VariantCallingSample::depth tensor is not defined!");
    }

    const int64_t num_columns = std::ssize(positions_major);

    if (logits.size(0) != num_columns) {
        std::ostringstream oss;
        oss << "VariantCallingSample::logits is of incorrect size. logits.size = " << logits.size(0)
            << ", num_columns = " << num_columns;
        throw std::runtime_error(oss.str());
    }

    if (depth.size(0) != num_columns) {
        std::ostringstream oss;
        oss << "VariantCallingSample::depth is of incorrect size. depth.size = " << depth.size(0)
            << ", num_columns = " << num_columns;
        throw std::runtime_error(oss.str());
    }

    // Basecalls may be absent; if present, they are [positions, reads, features].
    if (basecalls.defined()) {
        if (basecalls.dim() != 2) {
            std::ostringstream oss;
            oss << "VariantCallingSample::basecalls must be rank 2 with shape [positions, reads]. "
                   "Got shape = "
                << utils::tensor_shape_as_string(basecalls);
            throw std::runtime_error(oss.str());
        }

        if (basecalls.size(0) != num_columns) {
            std::ostringstream oss;
            oss << "VariantCallingSample::basecalls is of incorrect size. basecalls.shape = "
                << utils::tensor_shape_as_string(basecalls) << ", num_columns = " << num_columns;
            throw std::runtime_error(oss.str());
        }

        const int64_t num_reads = basecalls.size(1);
        if (std::ssize(read_ids_left) != num_reads) {
            std::ostringstream oss;
            oss << "VariantCallingSample::read_ids_left has incorrect size. read_ids_left.size = "
                << std::size(read_ids_left) << ", expected size = " << num_reads;
            throw std::runtime_error(oss.str());
        }

        if (std::ssize(read_ids_right) != num_reads) {
            std::ostringstream oss;
            oss << "VariantCallingSample::read_ids_right has incorrect size. read_ids_right.size = "
                << std::size(read_ids_right) << ", expected size = " << num_reads;
            throw std::runtime_error(oss.str());
        }
    }
}

int64_t VariantCallingSample::start() const {
    return (std::empty(positions_major) ? -1 : (positions_major.front()));
}

int64_t VariantCallingSample::end() const {
    return (std::empty(positions_major) ? -1 : (positions_major.back() + 1));
}

void debug_print(std::ostream& os,
                 const secondary::VariantCallingSample& vc_sample,
                 const bool detailed) {
    // Make sure that vectors are of the same length.
    vc_sample.validate();

    // Print scalar info.
    os << "seq_id = " << vc_sample.seq_id << ", positions = " << vc_sample.start() << " - "
       << vc_sample.end() << " , dist = " << (vc_sample.end() - vc_sample.start())
       << ", values = [";

    // Print first the beginning and end of the positions vectors.
    constexpr int64_t START = 0;
    constexpr int64_t MARGIN = 3;
    const int64_t len = std::ssize(vc_sample.positions_major);
    for (int64_t k = START; k < std::min<int64_t>(START + MARGIN, len); ++k) {
        os << "(" << vc_sample.positions_major[k] << ", " << vc_sample.positions_minor[k] << ") ";
        os.flush();
    }
    os << " ...";
    os.flush();
    const int64_t end = len;
    for (int64_t k = std::max<int64_t>(START + MARGIN, end - MARGIN); k < end; ++k) {
        os << " (" << vc_sample.positions_major[k] << ", " << vc_sample.positions_minor[k] << ")";
        os.flush();
    }
    os << "], size = " << std::size(vc_sample.positions_major);
    os << ", basecalls.size() = "
       << (vc_sample.basecalls.defined() ? vc_sample.basecalls.size(0) : 0);
    os.flush();

    if (detailed) {
        os << '\n';
        for (int64_t k = 0; k < len; ++k) {
            os << "[k = " << k << "] major = " << vc_sample.positions_major[k]
               << ", minor = " << vc_sample.positions_minor[k] << '\n';
        }
        os.flush();
    }
}

std::string to_string(const secondary::VariantCallingSample& vc_sample, const bool detailed) {
    std::ostringstream oss;
    secondary::debug_print(oss, vc_sample, detailed);
    return std::move(oss).str();
}

std::ostream& operator<<(std::ostream& os, const VariantCallingSample& vc_sample) {
    debug_print(os, vc_sample, false);
    return os;
}

bool operator==(const VariantCallingSample& lhs, const VariantCallingSample& rhs) {
    const auto bcs_equal = [&lhs, &rhs]() {
        if (lhs.basecalls.defined() != rhs.basecalls.defined()) {
            return false;
        }
        if (lhs.basecalls.defined()) {
            return lhs.basecalls.equal(rhs.basecalls);
        }
        return true;
    };
    return (lhs.logits.equal(rhs.logits)) &&
           (std::tie(lhs.seq_id, lhs.positions_major, lhs.positions_minor) ==
            std::tie(rhs.seq_id, rhs.positions_major, rhs.positions_minor)) &&
           (lhs.depth.equal(rhs.depth)) && bcs_equal();
}

namespace {
template <typename T>
inline T get_front_val(const std::vector<T>& data, const T default_val) {
    return std::empty(data) ? default_val : data.front();
}
template <typename T>
inline T get_back_val(const std::vector<T>& data, const T default_val) {
    return std::empty(data) ? default_val : data.back();
}

/**
 * \brief Returns true if the rhs sample directly continues the lhs sample: either the major coordinates
 *          are adjacent and rhs starts at minor zero, or the major coordinates are equal and the minor
 *          coordinates are adjacent.
 */
bool are_samples_adjacent(const VariantCallingSample& lhs, const VariantCallingSample& rhs) {
    if (std::empty(lhs.positions_major) || std::empty(rhs.positions_major)) {
        return false;
    }
    return (lhs.seq_id == rhs.seq_id) &&
           (((rhs.positions_major.front() == (lhs.positions_major.back() + 1)) &&
             (rhs.positions_minor.front() == 0)) ||
            ((rhs.positions_major.front() == lhs.positions_major.back()) &&
             (rhs.positions_minor.front() == (lhs.positions_minor.back() + 1))));
}
}  // namespace

bool variant_calling_sample_less(const VariantCallingSample& lhs, const VariantCallingSample& rhs) {
    const auto order_key = [](const VariantCallingSample& sample) {
        constexpr int64_t def = -1;
        const int64_t start_major = get_front_val(sample.positions_major, def);
        const int64_t start_minor = get_front_val(sample.positions_minor, def);
        const int64_t end_major = get_back_val(sample.positions_major, def);
        const int64_t end_minor = get_back_val(sample.positions_minor, def);
        const int64_t neg_len = -std::ssize(sample.positions_major);
        return std::tuple{sample.seq_id, start_major, start_minor, neg_len, end_major, end_minor};
    };

    return order_key(lhs) < order_key(rhs);
}

VariantCallingSample slice_vc_sample(const VariantCallingSample& vc_sample,
                                     const int64_t idx_start,
                                     const int64_t idx_end,
                                     const bool clone) {
    // Check that all members of the sample are of the same length.
    vc_sample.validate();

    const int64_t num_columns = std::ssize(vc_sample.positions_major);

    // Validate idx.
    if ((idx_start < 0) || (idx_start >= num_columns) || (idx_start >= idx_end) ||
        (idx_end > num_columns)) {
        throw std::out_of_range("Index is out of range in slice_vc_sample. idx_start = " +
                                std::to_string(idx_start) +
                                ", idx_end = " + std::to_string(idx_end) +
                                ", num_columns = " + std::to_string(num_columns));
    }

    std::vector<std::string> placeholder_rids =
            ((idx_start > 0) || (idx_end < num_columns))
                    ? placeholder_read_ids(std::ssize(vc_sample.read_ids_left))
                    : std::vector<std::string>{};

    // Slice.
    if (clone) {
        VariantCallingSample ret{
                .seq_id = vc_sample.seq_id,
                .positions_major =
                        std::vector<int64_t>(std::begin(vc_sample.positions_major) + idx_start,
                                             std::begin(vc_sample.positions_major) + idx_end),
                .positions_minor =
                        std::vector<int64_t>(std::begin(vc_sample.positions_minor) + idx_start,
                                             std::begin(vc_sample.positions_minor) + idx_end),
                .logits = vc_sample.logits.index({at::indexing::Slice(idx_start, idx_end)}).clone(),
                .depth = vc_sample.depth.index({at::indexing::Slice(idx_start, idx_end)}).clone(),
                .read_ids_left = (idx_start == 0) ? vc_sample.read_ids_left : placeholder_rids,
                .read_ids_right =
                        (idx_end == num_columns) ? vc_sample.read_ids_right : placeholder_rids,
                .basecalls = vc_sample.basecalls.defined()
                                     ? vc_sample.basecalls
                                               .index({at::indexing::Slice(idx_start, idx_end)})
                                               .clone()
                                     : at::Tensor{}};
        ret.validate();
        return ret;
    }

    VariantCallingSample ret{
            .seq_id = vc_sample.seq_id,
            .positions_major =
                    std::vector<int64_t>(std::begin(vc_sample.positions_major) + idx_start,
                                         std::begin(vc_sample.positions_major) + idx_end),
            .positions_minor =
                    std::vector<int64_t>(std::begin(vc_sample.positions_minor) + idx_start,
                                         std::begin(vc_sample.positions_minor) + idx_end),
            .logits = vc_sample.logits.index({at::indexing::Slice(idx_start, idx_end)}),
            .depth = vc_sample.depth.index({at::indexing::Slice(idx_start, idx_end)}),
            .read_ids_left = (idx_start == 0) ? vc_sample.read_ids_left : placeholder_rids,
            .read_ids_right =
                    (idx_end == num_columns) ? vc_sample.read_ids_right : placeholder_rids,
            .basecalls =
                    vc_sample.basecalls.defined()
                            ? vc_sample.basecalls.index({at::indexing::Slice(idx_start, idx_end)})
                            : at::Tensor{}};
    ret.validate();
    return ret;
}

VariantCallingSample slice_vc_sample_in_ref_coords(const VariantCallingSample& vc_sample,
                                                   const int64_t ref_start,
                                                   const int64_t ref_end) {
    // Check that all members of the sample are of the same length.
    vc_sample.validate();

    if (std::empty(vc_sample.positions_major)) {
        return vc_sample;
    }

    // Nothing to do.
    if ((ref_start == vc_sample.start()) && (ref_end == vc_sample.end())) {
        return vc_sample;
    }

    // Validate reference coords.
    if ((ref_start < 0) || (ref_end < 0) || (ref_start >= ref_end) ||
        (ref_start >= vc_sample.end()) || (ref_end > vc_sample.end())) {
        throw std::out_of_range(
                "Reference coordinates are out of range in slice_vc_sample_in_ref_coords. "
                "ref_start = " +
                std::to_string(ref_start) + ", ref_end = " + std::to_string(ref_end) +
                ", vc_sample.start() = " + std::to_string(vc_sample.start()) +
                ", vc_sample.end() = " + std::to_string(vc_sample.end()));
    }

    // Find where the reference coordinates actually begin in the sample.
    const auto start_iter = std::lower_bound(std::begin(vc_sample.positions_major),
                                             std::end(vc_sample.positions_major), ref_start);
    const auto end_iter = std::lower_bound(std::begin(vc_sample.positions_major),
                                           std::end(vc_sample.positions_major), ref_end);
    const int64_t start_idx = std::distance(std::begin(vc_sample.positions_major), start_iter);
    const int64_t end_idx = std::distance(std::begin(vc_sample.positions_major), end_iter);

    return slice_vc_sample(vc_sample, start_idx, end_idx, true);
}

std::vector<VariantCallingSample> merge_vc_samples(std::vector<VariantCallingSample> vc_samples) {
    // modified from secondary/features/encoder_read_alignment.cpp
    const auto cat_vectors_in_place = [](std::vector<int64_t>& v1, const std::vector<int64_t>& v2) {
        v1.reserve(std::size(v1) + std::size(v2));
        v1.insert(std::end(v1), std::cbegin(v2), std::cend(v2));
    };

    const auto pad_reads = [](std::vector<at::Tensor> chunks, int64_t target_depth) {
        // Determine the target depth if not provided
        if (target_depth < 0) {
            target_depth = 0;
            for (const auto& chunk : chunks) {
                target_depth = std::max(target_depth, chunk.size(1));
            }
        }

        bool needs_padding = false;
        for (const auto& chunk : chunks) {
            if (chunk.size(1) < target_depth) {
                needs_padding = true;
                break;
            }
        }
        if (!needs_padding) {
            return chunks;
        }

        // Pad each chunk to match the target depth
        std::vector<at::Tensor> padded_chunks;
        for (auto& chunk : chunks) {
            const int64_t pad_depth = target_depth - chunk.size(1);
            if (pad_depth > 0) {
                auto padding = at::zeros({chunk.size(0), pad_depth}, chunk.options());
                auto concated = at::cat({std::move(chunk), std::move(padding)}, 1);
                padded_chunks.emplace_back(std::move(concated));
            } else {
                padded_chunks.emplace_back(std::move(chunk));
            }
        }

        return padded_chunks;
    };

    /**
     * \brief This lambda reorders reads in the chunk tensors, because some rows (reads) may be missing
     *          between neighboring samples (e.g. some reads end/begin).
     */
    const auto reorder_reads = [](std::vector<at::Tensor> chunks,
                                  const std::vector<std::vector<std::string>>& read_ids_in,
                                  const std::vector<std::vector<std::string>>& read_ids_out) {
        if (std::size(chunks) < 2) {
            return std::tuple<std::vector<at::Tensor>, std::vector<std::string>,
                              std::vector<std::string>>{std::move(chunks), read_ids_in.front(),
                                                        read_ids_out.back()};
        }

        bool already_ordered = true;
        for (std::size_t n = 1; n < std::size(chunks); ++n) {
            if (read_ids_out[n - 1] != read_ids_in[n]) {
                already_ordered = false;
                break;
            }
        }
        if (already_ordered) {
            return std::tuple<std::vector<at::Tensor>, std::vector<std::string>,
                              std::vector<std::string>>{std::move(chunks), read_ids_in.front(),
                                                        read_ids_out.back()};
        }

        std::vector<at::Tensor> reordered_chunks;
        reordered_chunks.reserve(std::ssize(chunks));
        reordered_chunks.emplace_back(std::move(chunks.front()));

        std::vector<std::string> new_left_rids = read_ids_in.front();
        std::vector<std::string> new_right_rids = read_ids_out.front();

        for (std::size_t n = 1; n < std::size(chunks); ++n) {
            // reorder_chunk expects 3-dim inputs, so expand the dimension and then
            // squeeze again
            auto [reordered_chunk, next_rids_out] = reorder_chunk(
                    chunks[n].unsqueeze(-1), new_right_rids, read_ids_in[n], read_ids_out[n]);
            reordered_chunks.emplace_back(reordered_chunk.squeeze(-1));
            new_right_rids = std::move(next_rids_out);
        }

        if (std::size(new_left_rids) < std::size(new_right_rids)) {
            new_left_rids.resize(std::size(new_right_rids), std::string(BLANK_READ_PLACEHOLDER));
        }

        return std::tuple<std::vector<at::Tensor>, std::vector<std::string>,
                          std::vector<std::string>>{
                std::move(reordered_chunks), std::move(new_left_rids), std::move(new_right_rids)};
    };

    const auto merge_adjacent_samples_in_place = [&cat_vectors_in_place, &pad_reads,
                                                  &reorder_reads](VariantCallingSample& lh,
                                                                  const VariantCallingSample& rh) {
        // Insert positions vectors.
        cat_vectors_in_place(lh.positions_major, rh.positions_major);
        cat_vectors_in_place(lh.positions_minor, rh.positions_minor);
        // Merge the tensors.
        lh.logits = at::cat({std::move(lh.logits), rh.logits});
        lh.depth = at::cat({std::move(lh.depth), rh.depth});
        // Merge basecall arrays
        if (lh.basecalls.defined() != rh.basecalls.defined()) {
            throw std::runtime_error{
                    "Inconsistent basecall definitions when merging VariantCallingSamples"};
        }
        if (lh.basecalls.defined() && rh.basecalls.defined()) {
            auto [reordered_reads, reordered_read_ids_left, reordered_read_ids_right] =
                    reorder_reads({std::move(lh.basecalls), rh.basecalls},
                                  {std::move(lh.read_ids_left), rh.read_ids_left},
                                  {std::move(lh.read_ids_right), rh.read_ids_right});
            lh.basecalls = at::cat(pad_reads(std::move(reordered_reads), -1));
            lh.read_ids_left = std::move(reordered_read_ids_left);
            lh.read_ids_right = std::move(reordered_read_ids_right);
        }

        // Validate the result. This can throw.
        lh.validate();
    };

    if (std::empty(vc_samples)) {
        return {};
    }
    if (std::size(vc_samples) == 1) {
        vc_samples.front().validate();
        return vc_samples;
    }

    // Validate sample for sanity. This can throw.
    vc_samples[0].validate();
    std::vector<VariantCallingSample> ret;
    ret.reserve(std::size(vc_samples));
    ret.emplace_back(std::move(vc_samples.front()));

    for (std::size_t i = 1; i < std::size(vc_samples); ++i) {
        const VariantCallingSample& last = ret.back();
        VariantCallingSample& curr = vc_samples[i];

        // Validate sample for sanity. This can throw.
        curr.validate();

        // Should not happen, but sanity check.
        if (std::empty(last.positions_major) || std::empty(curr.positions_major)) {
            ret.emplace_back(std::move(curr));
            continue;
        }

        if (are_samples_adjacent(last, curr)) {
            merge_adjacent_samples_in_place(ret.back(), vc_samples[i]);
            curr = {};
        } else {
            ret.emplace_back(std::move(curr));
        }
    }

    return ret;
}

namespace {

/**
 * \brief Decodes the sample and flags the columns where any haplotype differs from the draft.
 * \returns Per-column variant flags, empty if the sample could not be decoded.
 */
std::vector<bool> find_variant_columns(const VariantCallingSample& vc_sample,
                                       const std::string& draft,
                                       const DecoderBase& decoder) {
    // Unsqueeze the logits because this vector contains logits for each individual sample of the shape
    // [positions x class_probabilities], whereas the decode_bases function expects that the first dimension is
    // the batch sample ID. That is, the tensor should be of shape: [batch_sample_id x positions x class_probabilities].
    // In this case, the "batch size" is 1.
    const at::Tensor logits = vc_sample.logits.unsqueeze(0);
    const std::vector<std::vector<ConsensusResult>> c = decoder.decode_bases(logits);

    // This shouldn't be possible.
    if ((std::size(c) != 1) || (std::empty(c.front()))) {
        DORADO_LOG(warn,
                   "Unexpected number of consensus sequences generated from a single sample: "
                   "c.size = {}, c[0].size = {}. Skipping consensus of this sample.",
                   std::size(c), (std::empty(c) ? -1 : std::size(c.front())));
        return {};
    }

    // Sequences for comparison.
    const std::string draft_with_gaps =
            extract_draft_with_gaps(draft, vc_sample.positions_major, vc_sample.positions_minor);

    // Use std::nullopt to be more strict about merging/breaking on variants.
    return find_polyploid_variants(vc_sample.positions_minor, draft_with_gaps, c.front(),
                                   std::nullopt);
}

}  // namespace

/**
 * \brief This function merges all neighboring VC sample windows, without splitting on variants.
 */
std::vector<VariantCallingSample> join_samples(std::vector<VariantCallingSample> vc_samples) {
    std::vector<VariantCallingSample> ret;

    // Holds the samples which continue each other, until they are merged and moved to ret.
    std::vector<VariantCallingSample> queue;

    // Merges the queued samples and moves them to the output.
    const auto flush_queue = [&ret, &queue]() {
        if (std::empty(queue)) {
            return;
        }
        auto new_samples = merge_vc_samples(std::move(queue));
        ret.insert(std::end(ret), std::make_move_iterator(std::begin(new_samples)),
                   std::make_move_iterator(std::end(new_samples)));
        queue.clear();
    };

    for (VariantCallingSample& sample : vc_samples) {
        sample.validate();

        // Enqueue the sample.
        if (!std::empty(queue) && !are_samples_adjacent(queue.back(), sample)) {
            flush_queue();
        }
        queue.emplace_back(std::move(sample));
    }

    flush_queue();

    return ret;
}

std::vector<VariantCallingSample> split_vc_samples_on_boundaries(
        std::vector<VariantCallingSample> vc_samples,
        const std::span<const int64_t> boundaries,
        const std::string& draft,
        const DecoderBase& decoder) {
    if (std::empty(boundaries)) {
        return vc_samples;
    }

    std::vector<VariantCallingSample> ret;
    ret.reserve(std::size(vc_samples));

    auto it_boundary = std::begin(boundaries);

    for (VariantCallingSample& vc_sample : vc_samples) {
        vc_sample.validate();

        if (std::empty(vc_sample.positions_major)) {
            ret.emplace_back(std::move(vc_sample));
            continue;
        }

        // Only split samples which have a boundary strictly inside.
        const int64_t sample_end = vc_sample.end();
        it_boundary = std::upper_bound(it_boundary, std::end(boundaries), vc_sample.start());
        if ((it_boundary == std::end(boundaries)) || (*it_boundary >= sample_end)) {
            ret.emplace_back(std::move(vc_sample));
            continue;
        }

        const std::vector<bool> is_variant = find_variant_columns(vc_sample, draft, decoder);
        if (std::empty(is_variant)) {
            ret.emplace_back(std::move(vc_sample));
            continue;
        }

        const std::vector<int64_t>& major = vc_sample.positions_major;
        const std::vector<int64_t>& minor = vc_sample.positions_minor;
        const int64_t num_positions = std::ssize(major);

        std::vector<int64_t> cuts;
        for (; (it_boundary != std::end(boundaries)) && (*it_boundary < sample_end);
             ++it_boundary) {
            const int64_t k = std::distance(
                    std::cbegin(major),
                    std::lower_bound(std::cbegin(major), std::cend(major), *it_boundary));
            if ((k <= 0) || (k >= num_positions)) {
                continue;
            }
            if (!(is_variant[k - 1] && is_variant[k])) {
                cuts.emplace_back(k);
                continue;
            }

            // The boundary is inside a variant. Defer the cut to the variant end, so that
            // normalisation of the variant start has its left context.
            int64_t start = k - 1;
            while ((start > 0) && is_variant[start - 1]) {
                --start;
            }
            // The variant touches the sample start, skip as no valid reference positions before split
            if (start == 0) {
                continue;
            }
            int64_t end = k;
            while ((end < num_positions) && (is_variant[end] || (minor[end] != 0))) {
                ++end;
            }
            // The variant touches the sample end, skip as no valid reference positions after the split
            if (end >= num_positions) {
                continue;
            }
            cuts.emplace_back(end);
        }

        std::sort(std::begin(cuts), std::end(cuts));
        cuts.erase(std::unique(std::begin(cuts), std::end(cuts)), std::end(cuts));

        if (std::empty(cuts)) {
            ret.emplace_back(std::move(vc_sample));
            continue;
        }

        int64_t prev = 0;
        for (const int64_t cut : cuts) {
            ret.emplace_back(slice_vc_sample(vc_sample, prev, cut, true));
            prev = cut;
        }
        ret.emplace_back(slice_vc_sample(vc_sample, prev, num_positions, true));
    }

    return ret;
}

std::vector<VariantCallingSample> trim_vc_samples(
        const std::vector<VariantCallingSample>& vc_input_data,
        const std::vector<int32_t>& ordered_ids) {
    // Mock the Sample objects. Trimming works on Sample objects only, but
    // it only needs positions, not the actual tensors.
    std::vector<Sample> local_samples;
    local_samples.reserve(std::size(ordered_ids));
    for (const int32_t id : ordered_ids) {
        const auto& vc_sample = vc_input_data[id];
        local_samples.emplace_back(Sample{.seq_id = vc_sample.seq_id,
                                          .features = {},
                                          .positions_major = vc_sample.positions_major,
                                          .positions_minor = vc_sample.positions_minor,
                                          .depth = vc_sample.depth,
                                          .read_ids_left = vc_sample.read_ids_left,
                                          .read_ids_right = vc_sample.read_ids_right});
    }

    // Compute trimming of all samples for this group.
    const std::vector<TrimInfo> trims = trim_samples(local_samples, std::nullopt);

    std::vector<VariantCallingSample> trimmed_samples;

    assert(std::size(trims) == std::size(local_samples));
    assert(std::size(trims) == std::size(ordered_ids));

    for (std::size_t i = 0; i < std::size(trims); ++i) {
        const int32_t id = ordered_ids[i];
        const auto& s = vc_input_data[id];
        const TrimInfo& t = trims[i];

        // Make sure that all vectors and tensors are of the same length.
        try {
            s.validate();
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << "Sample not valid in trim_vc_samples! Skipping the sample. Sample: " << s
                << ", trim: " << t << ", exception: '" << e.what();
            DORADO_LOG(warn, oss.str());
            continue;
        }

        // Skip samples which were filtered during by trimming (coords are
        // out of bounds or not valid).
        if (!is_trim_info_valid(t, std::ssize(s.positions_major))) {
            continue;
        }

        // Avoid cloning tensor storage when trimming keeps the complete sample.
        if ((t.start == 0) && (t.end == std::ssize(s.positions_major))) {
            trimmed_samples.emplace_back(s);
        } else {
            trimmed_samples.emplace_back(slice_vc_sample(s, t.start, t.end, true));
        }
    }

    return trimmed_samples;
}

}  // namespace dorado::secondary
