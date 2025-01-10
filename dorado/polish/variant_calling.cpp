#include "variant_calling.h"

#include "utils/ssize.h"

#include <spdlog/spdlog.h>

namespace dorado::polisher {

std::vector<std::string> call_variants(
        const dorado::polisher::Interval& region_batch,
        const VariantCallingInputData& vc_input_data,
        const hts_io::FastxRandomReader& /*draft_reader*/,
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

    return {};
}

}  // namespace dorado::polisher
