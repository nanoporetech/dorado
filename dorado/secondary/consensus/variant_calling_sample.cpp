#include "secondary/consensus/variant_calling_sample.h"

#include "secondary/consensus/consensus_utils.h"
#include "secondary/consensus/sample.h"
#include "secondary/consensus/sample_trimming.h"
#include "secondary/features/decoder_base.h"

#include <spdlog/spdlog.h>

#include <algorithm>
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
    return (lhs.logits.equal(rhs.logits)) &&
           (std::tie(lhs.seq_id, lhs.positions_major, lhs.positions_minor) ==
            std::tie(rhs.seq_id, rhs.positions_major, rhs.positions_minor)) &&
           (lhs.depth.equal(rhs.depth));
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
                                     const int64_t idx_end) {
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

    // Slice.
    return VariantCallingSample{
            vc_sample.seq_id,
            std::vector<int64_t>(std::begin(vc_sample.positions_major) + idx_start,
                                 std::begin(vc_sample.positions_major) + idx_end),
            std::vector<int64_t>(std::begin(vc_sample.positions_minor) + idx_start,
                                 std::begin(vc_sample.positions_minor) + idx_end),
            vc_sample.logits.index({at::indexing::Slice(idx_start, idx_end)}).clone(),
            vc_sample.depth.index({at::indexing::Slice(idx_start, idx_end)}).clone()};
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

    return slice_vc_sample(vc_sample, start_idx, end_idx);
}

std::vector<VariantCallingSample> merge_vc_samples(std::vector<VariantCallingSample> vc_samples) {
    const auto merge_adjacent_samples_in_place = [](VariantCallingSample& lh,
                                                    const VariantCallingSample& rh) {
        const size_t width = std::size(lh.positions_major);

        // Insert positions vectors.
        lh.positions_major.reserve(width + std::size(rh.positions_major));
        lh.positions_major.insert(std::end(lh.positions_major), std::begin(rh.positions_major),
                                  std::end(rh.positions_major));
        lh.positions_minor.reserve(width + std::size(rh.positions_minor));
        lh.positions_minor.insert(std::end(lh.positions_minor), std::begin(rh.positions_minor),
                                  std::end(rh.positions_minor));

        // Merge the tensors.
        lh.logits = at::cat({std::move(lh.logits), rh.logits});
        lh.depth = at::cat({std::move(lh.depth), rh.depth});
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

        // Merge if major coordinates are adjacent and the next sample starts at minor zero,
        // or if major coordinates are equal but minor coordinates are adjacent.
        if ((curr.seq_id == last.seq_id) &&
            (((curr.positions_major[0] == (last.positions_major.back() + 1)) &&
              (curr.positions_minor[0] == 0)) ||
             ((curr.positions_major[0] == last.positions_major.back()) &&
              (curr.positions_minor[0] == (last.positions_minor.back() + 1))))) {
            merge_adjacent_samples_in_place(ret.back(), vc_samples[i]);
            curr = {};
        } else {
            ret.emplace_back(std::move(curr));
        }
    }

    return ret;
}

/**
 * \brief This function restructures the neighboring samples for one draft sequence.
 *          Each input sample is split on the last non-variant position.
 *          The left part is merged with anything previously added to a queue (i.e.
 *          previous sample's right portion). The right part is then added to a cleared queue.
 *          The goal of this function is to prevent calling variants on sample boundaries.
 */
std::vector<VariantCallingSample> join_samples(std::vector<VariantCallingSample> vc_samples,
                                               const std::string& draft,
                                               const DecoderBase& decoder) {
    std::vector<VariantCallingSample> ret;

    std::vector<VariantCallingSample> queue;

    for (std::size_t i = 0; i < std::size(vc_samples); ++i) {
        VariantCallingSample& vc_sample = vc_samples[i];

        vc_sample.validate();

        // Unsqueeze the logits because this vector contains logits for each individual sample of the shape
        // [positions x class_probabilities], whereas the decode_bases function expects that the first dimension is
        // the batch sample ID. That is, the tensor should be of shape: [batch_sample_id x positions x class_probabilities].
        // In this case, the "batch size" is 1.
        const at::Tensor logits = vc_sample.logits.unsqueeze(0);
        const std::vector<std::vector<ConsensusResult>> c = decoder.decode_bases(logits);

        // This shouldn't be possible.
        if ((std::size(c) != 1) || (std::empty(c.front()))) {
            spdlog::warn(
                    "Unexpected number of consensus sequences generated from a single sample: "
                    "c.size = {}, c[0].size = {}. Skipping consensus of this sample.",
                    std::size(c), (std::empty(c) ? -1 : std::size(c.front())));
            continue;
        }

        // Sequences for comparison.
        const std::string draft_with_gaps = extract_draft_with_gaps(
                draft, vc_sample.positions_major, vc_sample.positions_minor);
        const std::vector<ConsensusResult>& calls_with_gaps = c.front();

        // Use std::nullopt to be more strict about merging/breaking on variants.
        const std::vector<bool> is_variant = find_polyploid_variants(
                vc_sample.positions_minor, draft_with_gaps, calls_with_gaps, std::nullopt);

        const int64_t num_positions = std::ssize(vc_sample.positions_major);

        // Find a location where to split the sample.
        int64_t last_non_var_start = 0;
        for (int64_t j = (num_positions - 1); j >= 0; --j) {
            if ((vc_sample.positions_minor[j] == 0) && !is_variant[j]) {
                last_non_var_start = j;
                break;
            }
        }

        // All positions differ. Merge the entire sample with the next one.
        if (last_non_var_start <= 0) {
            queue.emplace_back(std::move(vc_sample));
            continue;
        }

        // Split the sample.
        VariantCallingSample left_slice = slice_vc_sample(vc_sample, 0, last_non_var_start);
        VariantCallingSample right_slice =
                slice_vc_sample(vc_sample, last_non_var_start, num_positions);
        vc_sample = {};

        // Enqueue the queue if possible.
        if (last_non_var_start > 0) {
            queue.emplace_back(std::move(left_slice));
        }

        // Merge and insert.
        if (!std::empty(queue)) {
            auto new_samples = merge_vc_samples(std::move(queue));

            ret.insert(std::end(ret), std::make_move_iterator(std::begin(new_samples)),
                       std::make_move_iterator(std::end(new_samples)));
        }

        // Reset the queue.
        queue = {std::move(right_slice)};
    }

    // Merge and insert.
    if (!std::empty(queue)) {
        auto new_samples = merge_vc_samples(std::move(queue));

        ret.insert(std::end(ret), std::make_move_iterator(std::begin(new_samples)),
                   std::make_move_iterator(std::end(new_samples)));
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
                                          .read_ids_left = {},
                                          .read_ids_right = {}});
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
            spdlog::warn(oss.str());
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
            trimmed_samples.emplace_back(slice_vc_sample(s, t.start, t.end));
        }
    }

    return trimmed_samples;
}

}  // namespace dorado::secondary
