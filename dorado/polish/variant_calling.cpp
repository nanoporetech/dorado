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
std::string create_draft_with_gaps(const std::string& draft,
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

/**
 * \brief This function restructures the neighboring samples for one draft sequence
 */
std::vector<Sample> join_samples(
        const std::vector<VariantCallingSample>& vc_input_data,  // All samples for all drafts.
        const std::vector<std::pair<int64_t, int32_t>>&
                group_info,  // Samples which belong to the current draft, sorted by start coord.
        // const std::vector<const Sample*>& samples,
        const std::vector<TrimInfo>& trims,  // Trimming coordinates for these samples.
        const std::string& draft,
        const DecoderBase& decoder  // Decoder for logits -> bases.
) {
    // std::vector<ConsensusResult> consensuses(std::size(group_info));
    // Compute the untrimmed consensus of each sample.

    std::vector<torch::Tensor> queue;

    for (int64_t i = 0; i < dorado::ssize(group_info); ++i) {
        const auto& [start, id] = group_info[i];

        // Unsqueeze the logits because this vector contains logits for each individual sample of the shape
        // [positions x class_probabilities], whereas the decode_bases function expects that the first dimension is
        // the batch sample ID. That is, the tensor should be of shape: [batch_sample_id x positions x class_probabilities].
        // In this case, the "batch size" is 1.
        const at::Tensor logits = vc_input_data[id].logits.unsqueeze(0);
        std::vector<ConsensusResult> c = decoder.decode_bases(logits);

        if (std::size(c) != 1) {
            spdlog::warn(
                    "Unexpected number of consensus sequences generated from a single sample: "
                    "c.size = {}. Skipping consensus of this sample.",
                    std::size(c));
            continue;
        }

        const Sample& sample = vc_input_data[id].sample;

        const std::string& call_with_gaps = c.front().seq;
        const std::string draft_with_gaps =
                create_draft_with_gaps(draft, sample.positions_major, sample.positions_minor);

        assert(std::size(call_with_gaps) == std::size(draft_with_gaps));

        // Check if all positions are diffs, or if all positions are gaps in both sequences.
        // If so, merge the entire sample with the next one.
        int64_t count = 0;
        for (int64_t j = 0; j < dorado::ssize(call_with_gaps); ++j) {
            if ((call_with_gaps[j] != draft_with_gaps[j]) ||
                (call_with_gaps[j] == '*' && draft_with_gaps[j] == '*')) {
                ++count;
            }
        }
        if (count == dorado::ssize(call_with_gaps)) {
            // queue.emplace_back(...);
        }

        // consensuses[i] = std::move(c.front());

        // std::cout << "[i = " << i << "] sample = " << sample << "\nC: " << consensuses[i].seq << "\nD: " << draft << "\n";
    }

    (void)trims;

    return {};
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
        const auto trims = trim_samples(local_samples, std::nullopt);

        const auto joined_samples = join_samples(vc_input_data, group, trims, draft, decoder);

        // TODO:
        //      join_samples();
        //      decode_variants();
    }

    return {};
}

}  // namespace dorado::polisher
