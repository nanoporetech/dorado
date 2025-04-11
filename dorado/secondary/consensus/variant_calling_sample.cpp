#include "variant_calling_sample.h"

#include "consensus_utils.h"
#include "sample.h"
#include "sample_trimming.h"
#include "secondary/features/decoder_base.h"
#include "utils/ssize.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>

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

    const int64_t num_columns = dorado::ssize(positions_major);

    if (logits.size(0) != num_columns) {
        std::ostringstream oss;
        oss << "VariantCallingSample::logits is of incorrect size. logits.size = " << logits.size(0)
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

std::ostream& operator<<(std::ostream& os, const VariantCallingSample& vc_sample) {
    // Make sure that vectors are of the same length.
    vc_sample.validate();

    // Print scalar info.
    os << "seq_id = " << vc_sample.seq_id << ", positions = " << vc_sample.start() << " - "
       << vc_sample.end() << " , dist = " << (vc_sample.end() - vc_sample.start())
       << ", values = [";

    // Print first the beginning and end of the positions vectors.
    constexpr int64_t START = 0;
    const int64_t len = dorado::ssize(vc_sample.positions_major);
    for (int64_t k = START; k < std::min<int64_t>(START + 3, len); ++k) {
        os << "(" << vc_sample.positions_major[k] << ", " << vc_sample.positions_minor[k] << ") ";
        os.flush();
    }
    os << " ...";
    os.flush();
    const int64_t end = len;
    for (int64_t k = std::max<int64_t>(START + 3, end - 3); k < end; ++k) {
        os << " (" << vc_sample.positions_major[k] << ", " << vc_sample.positions_minor[k] << ")";
        os.flush();
    }
    os << "], size = " << std::size(vc_sample.positions_major);
    os.flush();

    return os;
}

bool operator==(const VariantCallingSample& lhs, const VariantCallingSample& rhs) {
    return (lhs.logits.equal(rhs.logits)) &&
           (std::tie(lhs.seq_id, lhs.positions_major, lhs.positions_minor) ==
            std::tie(rhs.seq_id, rhs.positions_major, rhs.positions_minor));
}

VariantCallingSample slice_vc_sample(const VariantCallingSample& vc_sample,
                                     const int64_t idx_start,
                                     const int64_t idx_end) {
    // Check that all members of the sample are of the same length.
    vc_sample.validate();

    const int64_t num_columns = dorado::ssize(vc_sample.positions_major);

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
            vc_sample.logits.index({at::indexing::Slice(idx_start, idx_end)}).clone()};
}

std::vector<VariantCallingSample> merge_vc_samples(
        const std::vector<VariantCallingSample>& vc_samples) {
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
        lh.logits = torch::cat({std::move(lh.logits), rh.logits});
    };

    if (std::empty(vc_samples)) {
        return {};
    }

    std::vector<VariantCallingSample> ret{vc_samples.front()};

    // Validate sample for sanity. This can throw.
    vc_samples[0].validate();

    for (int64_t i = 1; i < dorado::ssize(vc_samples); ++i) {
        const VariantCallingSample& last = ret.back();
        const VariantCallingSample& curr = vc_samples[i];

        // Validate sample for sanity. This can throw.
        curr.validate();

        // Should not happen, but sanity check.
        if (std::empty(last.positions_major) || std::empty(curr.positions_major)) {
            ret.emplace_back(vc_samples[i]);
            continue;
        }

        // Merge if major coordinates are adjacent, or if major coordinates are equal but minor are adjacent.
        if ((curr.seq_id == last.seq_id) &&
            ((curr.positions_major[0] == (last.positions_major.back() + 1)) ||
             ((curr.positions_major[0] == last.positions_major.back()) &&
              (curr.positions_minor[0] == (last.positions_minor.back() + 1))))) {
            merge_adjacent_samples_in_place(ret.back(), vc_samples[i]);
        } else {
            ret.emplace_back(vc_samples[i]);
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
std::vector<VariantCallingSample> join_samples(const std::vector<VariantCallingSample>& vc_samples,
                                               const std::string& draft,
                                               const DecoderBase& decoder) {
    std::vector<VariantCallingSample> ret;

    std::vector<VariantCallingSample> queue;

    for (int64_t i = 0; i < dorado::ssize(vc_samples); ++i) {
        const VariantCallingSample& vc_sample = vc_samples[i];

        vc_sample.validate();

        // Unsqueeze the logits because this vector contains logits for each individual sample of the shape
        // [positions x class_probabilities], whereas the decode_bases function expects that the first dimension is
        // the batch sample ID. That is, the tensor should be of shape: [batch_sample_id x positions x class_probabilities].
        // In this case, the "batch size" is 1.
        const at::Tensor logits = vc_sample.logits.unsqueeze(0);
        const std::vector<ConsensusResult> c = decoder.decode_bases(logits);

        // This shouldn't be possible.
        if (std::size(c) != 1) {
            spdlog::warn(
                    "Unexpected number of consensus sequences generated from a single sample: "
                    "c.size = {}. Skipping consensus of this sample.",
                    std::size(c));
            continue;
        }

        // Sequences for comparison.
        const std::string& call_with_gaps = c.front().seq;
        const std::string draft_with_gaps = extract_draft_with_gaps(
                draft, vc_sample.positions_major, vc_sample.positions_minor);
        assert(std::size(call_with_gaps) == std::size(draft_with_gaps));

        const auto check_is_diff = [](const char base1, const char base2) {
            return (base1 != base2) || (base1 == '*' && base2 == '*');
        };

        // Check if all positions are diffs, or if all positions are gaps in both sequences.
        {
            int64_t count = 0;
            for (int64_t j = 0; j < dorado::ssize(call_with_gaps); ++j) {
                if (check_is_diff(call_with_gaps[j], draft_with_gaps[j])) {
                    ++count;
                }
            }
            if (count == dorado::ssize(call_with_gaps)) {
                // Merge the entire sample with the next one. We need at least one non-diff non-gap pos.
                queue.emplace_back(vc_sample);
                continue;
            }
        }

        const int64_t num_positions = dorado::ssize(vc_sample.positions_major);

        // Find a location where to split the sample.
        int64_t last_non_var_start = 0;
        for (int64_t j = (num_positions - 1); j >= 0; --j) {
            if ((vc_sample.positions_minor[j] == 0) &&
                !check_is_diff(call_with_gaps[j], draft_with_gaps[j])) {
                last_non_var_start = j;
                break;
            }
        }

        // Split the sample.
        VariantCallingSample left_slice = slice_vc_sample(vc_sample, 0, last_non_var_start);
        VariantCallingSample right_slice =
                slice_vc_sample(vc_sample, last_non_var_start, num_positions);

        // Enqueue the queue if possible.
        if (last_non_var_start > 0) {
            queue.emplace_back(std::move(left_slice));
        }

        // Merge and insert.
        if (!std::empty(queue)) {
            auto new_samples = merge_vc_samples(queue);
            queue.clear();

            ret.insert(std::end(ret), std::make_move_iterator(std::begin(new_samples)),
                       std::make_move_iterator(std::end(new_samples)));
        }

        // Reset the queue.
        queue = {std::move(right_slice)};
    }

    // Merge and insert.
    if (!std::empty(queue)) {
        auto new_samples = merge_vc_samples(queue);
        queue.clear();

        ret.insert(std::end(ret), std::make_move_iterator(std::begin(new_samples)),
                   std::make_move_iterator(std::end(new_samples)));
    }

    return ret;
}

std::vector<VariantCallingSample> trim_vc_samples(
        const std::vector<VariantCallingSample>& vc_input_data,
        const std::vector<std::pair<int64_t, int32_t>>& group) {
    // Mock the Sample objects. Trimming works on Sample objects only, but
    // it only needs positions, not the actual tensors.
    std::vector<Sample> local_samples;
    local_samples.reserve(std::size(group));
    for (const auto& [start, id] : group) {
        const auto& vc_sample = vc_input_data[id];
        local_samples.emplace_back(Sample(vc_sample.seq_id, {}, vc_sample.positions_major,
                                          vc_sample.positions_minor, {}, {}, {}));
    }

    // Compute trimming of all samples for this group.
    const std::vector<TrimInfo> trims = trim_samples(local_samples, std::nullopt);

    std::vector<VariantCallingSample> trimmed_samples;

    assert(std::size(trims) == std::size(local_samples));
    assert(std::size(trims) == std::size(group));

    for (int64_t i = 0; i < dorado::ssize(trims); ++i) {
        const int32_t id = group[i].second;
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
        if (!is_trim_info_valid(t, dorado::ssize(s.positions_major))) {
            continue;
        }

        // Add a new trimmed sample.
        trimmed_samples.emplace_back(VariantCallingSample{
                s.seq_id,
                std::vector<int64_t>(std::begin(s.positions_major) + t.start,
                                     std::begin(s.positions_major) + t.end),
                std::vector<int64_t>(std::begin(s.positions_minor) + t.start,
                                     std::begin(s.positions_minor) + t.end),
                s.logits.index({at::indexing::Slice(t.start, t.end)}).clone()});
    }

    return trimmed_samples;
}

}  // namespace dorado::secondary
