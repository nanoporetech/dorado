#include "variant_calling.h"

#include "trim.h"
#include "utils/ssize.h"

#include <spdlog/spdlog.h>

namespace dorado::polisher {

std::vector<std::string> call_variants(
        const dorado::polisher::Interval& region_batch,
        const VariantCallingInputData& vc_input_data,
        const hts_io::FastxRandomReader& draft_reader,
        const std::vector<std::pair<std::string, int64_t>>& draft_lens) {
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

        // Create local contiguous samples for trimming.
        std::vector<const Sample*> local_samples;
        local_samples.reserve(std::size(group));
        // NOTE: I wouldn't use a reference here because both start and id are POD, but Clang complains.
        for (const auto& [start, id] : group) {
            local_samples.emplace_back(&(vc_input_data[id].first));
        }

        // Compute trimming of all samples for this draft sequence.
        const auto trims = trim_samples(local_samples, std::nullopt);
    }

    return {};
}

}  // namespace dorado::polisher
