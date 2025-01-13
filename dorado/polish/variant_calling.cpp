#include "variant_calling.h"

#include "consensus_result.h"
#include "trim.h"
#include "utils/ssize.h"

#include <spdlog/spdlog.h>

#include <cassert>

namespace dorado::polisher {

/**
 * \brief Copy the draft sequence for a given sample, and expand it with '*' in places of gaps.
 */
std::string extract_draft_with_gaps(const std::string& draft,
                                    const std::vector<int64_t>& positions_major,
                                    const std::vector<int64_t>& positions_minor) {
    if (std::size(positions_major) != std::size(positions_minor)) {
        throw std::runtime_error(
                "The positions_major and positions_minor are not of the same size! "
                "positions_major.size = " +
                std::to_string(std::size(positions_major)) +
                ", positions_minor.size = " + std::to_string(std::size(positions_minor)));
    }

    std::string ret(std::size(positions_major), '*');

    for (int64_t i = 0; i < dorado::ssize(positions_major); ++i) {
        ret[i] = (positions_minor[i] == 0) ? draft[positions_major[i]] : '*';
    }

    return ret;
}

std::string extract_draft(const std::string& draft, const std::vector<int64_t>& positions_major) {
    std::string ret(std::size(positions_major), '*');
    for (int64_t i = 0; i < dorado::ssize(positions_major); ++i) {
        ret[i] = draft[positions_major[i]];
    }
    return ret;
}

VariantCallingSample slice_vc_sample(const VariantCallingSample& vc_sample,
                                     const int64_t idx_start,
                                     const int64_t idx_end) {
    const int64_t num_columns = dorado::ssize(vc_sample.sample.positions_major);

    // Validate lengths.
    if (vc_sample.logits.defined() &&
        (vc_sample.logits.size(0) != dorado::ssize(vc_sample.sample.positions_major))) {
        throw std::runtime_error(
                "VariantCallingSample::logits is of incorrect size. logits.size = " +
                std::to_string(vc_sample.logits.size(0)) +
                ", num_columns = " + std::to_string(num_columns));
    }

    // Validate idx.
    if ((idx_start < 0) || (idx_start >= num_columns) || (idx_end >= idx_start) ||
        (idx_end > num_columns)) {
        throw std::out_of_range("Index is out of range in slice_vc_sample. idx_start = " +
                                std::to_string(idx_start) +
                                ", idx_end = " + std::to_string(idx_end) +
                                ", num_columns = " + std::to_string(num_columns));
    }

    // Slice.
    return VariantCallingSample{
            slice_sample(vc_sample.sample, idx_start, idx_end),
            vc_sample.logits.index({at::indexing::Slice(idx_start, idx_end)}).clone()};
}

std::vector<VariantCallingSample> merge_vc_samples(
        const std::vector<VariantCallingSample>& vc_samples) {
    if (std::empty(vc_samples)) {
        return {};
    }

    std::vector<VariantCallingSample> ret;

    ret.emplace_back(vc_samples.front());

    for (int64_t i = 1; i < dorado::ssize(vc_samples); ++i) {
        if (ret.back().sample.end() == (vc_samples[i].sample.start() + 1)) {
            merge_adjacent_samples_in_place(ret.back().sample, vc_samples[i].sample);
            ret.back().logits = torch::cat({std::move(ret.back().logits), vc_samples[i].logits});
        } else {
            ret.emplace_back(vc_samples[i]);
        }
    }

    return ret;
}

/**
 * \brief This function restructures the neighboring samples for one draft sequence
 */
std::vector<VariantCallingSample> join_samples(const std::vector<VariantCallingSample>& vc_samples,
                                               const std::string& draft,
                                               const DecoderBase& decoder) {
    std::vector<VariantCallingSample> ret;

    std::vector<VariantCallingSample> queue;

    for (int64_t i = 0; i < dorado::ssize(vc_samples); ++i) {
        const VariantCallingSample& vc_sample = vc_samples[i];
        const Sample& sample = vc_sample.sample;

        // Validate the sample.
        sample.validate();

        // Validate the logits
        if (!vc_sample.logits.defined()) {
            throw std::runtime_error("Logits tensor is not defined!");
        }
        if (vc_sample.logits.size(0) != dorado::ssize(vc_sample.sample.positions_major)) {
            throw std::runtime_error(
                    "Length of the logits tensor does not match sample length! logits.size = " +
                    std::to_string(vc_sample.logits.size(0)) + ", positions_major.size = " +
                    std::to_string(std::size(vc_sample.sample.positions_major)));
        }

        // Unsqueeze the logits because this vector contains logits for each individual sample of the shape
        // [positions x class_probabilities], whereas the decode_bases function expects that the first dimension is
        // the batch sample ID. That is, the tensor should be of shape: [batch_sample_id x positions x class_probabilities].
        // In this case, the "batch size" is 1.
        const at::Tensor logits = vc_sample.logits.unsqueeze(0);
        std::vector<ConsensusResult> c = decoder.decode_bases(logits);

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
        const std::string draft_with_gaps =
                extract_draft_with_gaps(draft, sample.positions_major, sample.positions_minor);
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

        const int64_t num_positions = dorado::ssize(sample.positions_major);

        // Find a location where to split the sample.
        int64_t last_non_var_start = 0;
        for (int64_t j = (num_positions - 1); j >= 0; --j) {
            if ((sample.positions_minor[j] == 0) &&
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
            ret.insert(std::end(ret), std::make_move_iterator(std::begin(new_samples)),
                       std::make_move_iterator(std::end(new_samples)));
        }

        // Reset the queue.
        queue = {std::move(right_slice)};
    }

    // Merge and insert.
    if (!std::empty(queue)) {
        auto new_samples = merge_vc_samples(queue);
        ret.insert(std::end(ret), std::make_move_iterator(std::begin(new_samples)),
                   std::make_move_iterator(std::end(new_samples)));
        queue.clear();
    }

    return ret;
}

std::vector<Sample> apply_trimming(const std::vector<const Sample*>& samples,
                                   const std::vector<TrimInfo>& trims) {
    std::vector<Sample> ret;

    for (int64_t i = 0; i < dorado::ssize(trims); ++i) {
        const Sample* s = samples[i];
        const TrimInfo& t = trims[i];
        ret.emplace_back(slice_sample(*s, t.start, t.end));
    }

    return ret;
}

std::vector<std::string> call_variants(
        const dorado::polisher::Interval& region_batch,
        const std::vector<VariantCallingSample>& vc_input_data,
        const hts_io::FastxRandomReader& draft_reader,
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        const DecoderBase& decoder) {
    // Group samples by sequence ID.
    std::vector<std::vector<std::pair<int64_t, int32_t>>> groups(region_batch.length());
    for (int32_t i = 0; i < dorado::ssize(vc_input_data); ++i) {
        const auto& [sample, logits] = vc_input_data[i];

        const int32_t local_id = sample.seq_id - region_batch.start;

        // Skip filtered samples.
        if (sample.seq_id < 0) {
            continue;
        }

        if ((sample.seq_id >= dorado::ssize(draft_lens)) || (local_id < 0) ||
            (local_id >= dorado::ssize(groups))) {
            spdlog::error(
                    "Draft ID out of bounds! r.draft_id = {}, draft_lens.size = {}, "
                    "groups.size = {}",
                    sample.seq_id, std::size(draft_lens), std::size(groups));
            continue;
        }
        groups[local_id].emplace_back(sample.start(), i);
    }

    // For each sequence, call variants.
    for (int64_t group_id = 0; group_id < dorado::ssize(groups); ++group_id) {
        const int64_t seq_id = group_id + region_batch.start;
        const std::string& header = draft_lens[seq_id].first;

        // Sort the group by start positions.
        auto& group = groups[group_id];
        std::sort(std::begin(group), std::end(group));

        if (std::empty(group)) {
            continue;
        }

        // Get the draft sequence.
        const std::string draft = draft_reader.fetch_seq(header);

        // Create a view into samples for this draft.
        std::vector<const Sample*> local_samples;
        local_samples.reserve(std::size(group));
        // NOTE: I wouldn't use a reference here because both start and id are POD, but Clang complains.
        for (const auto& [start, id] : group) {
            local_samples.emplace_back(&(vc_input_data[id].sample));
        }

        // Compute trimming of all samples for this draft sequence.
        const std::vector<TrimInfo> trims = trim_samples(local_samples, std::nullopt);

        // Produce trimmed samples.
        std::vector<Sample> trimmed_samples = apply_trimming(local_samples, trims);

        // Produce trimmed logits.
        std::vector<at::Tensor> trimmed_logits = [&]() {
            std::vector<at::Tensor> ret;
            for (size_t i = 0; i < std::size(trims); ++i) {
                const int64_t id = group[i].second;
                const TrimInfo& trim = trims[i];
                at::Tensor t = vc_input_data[id]
                                       .logits.index({at::indexing::Slice(trim.start, trim.end)})
                                       .clone();
                ret.emplace_back(std::move(t));
            }
            return ret;
        }();

        assert(std::size(trimmed_samples) == std::size(trimmed_logits));

        // Interleave the samples and logits for easier handling.
        const std::vector<VariantCallingSample> trimmed_vc_samples = [&]() {
            std::vector<VariantCallingSample> ret;
            for (size_t i = 0; i < std::size(trimmed_samples); ++i) {
                ret.emplace_back(VariantCallingSample{std::move(trimmed_samples[i]),
                                                      std::move(trimmed_logits[i])});
            }
            return ret;
        }();

        // Break and merge samples on non-variant positions.
        // const auto joined_samples = join_samples(vc_input_data, group, trims, draft, decoder);
        const auto joined_samples = join_samples(trimmed_vc_samples, draft, decoder);

        // TODO:
        //      join_samples();
        //      decode_variants();
    }

    return {};
}

}  // namespace dorado::polisher
