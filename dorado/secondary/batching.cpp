#include "batching.h"

#include <stdexcept>
#include <unordered_map>

namespace dorado::secondary {

std::pair<std::vector<std::vector<secondary::Region>>, std::vector<polisher::Interval>>
prepare_region_batches(const std::vector<std::pair<std::string, int64_t>>& draft_lens,
                       const std::vector<secondary::Region>& user_regions,
                       const int64_t draft_batch_size) {
    // Create a lookup.
    std::unordered_map<std::string, int64_t> draft_ids;
    for (int64_t seq_id = 0; seq_id < dorado::ssize(draft_lens); ++seq_id) {
        draft_ids[draft_lens[seq_id].first] = seq_id;
    }

    // Outer vector: ID of the draft, inner vector: regions.
    std::vector<std::vector<secondary::Region>> ret(std::size(draft_lens));

    if (std::empty(user_regions)) {
        // Add full draft sequences.
        for (int64_t seq_id = 0; seq_id < dorado::ssize(draft_lens); ++seq_id) {
            const auto& [draft_name, draft_len] = draft_lens[seq_id];
            ret[seq_id].emplace_back(secondary::Region{draft_name, 0, draft_len});
        }

    } else {
        // Bin the user regions for individual contigs.
        for (const auto& region : user_regions) {
            const auto it = draft_ids.find(region.name);
            if (it == std::end(draft_ids)) {
                throw std::runtime_error(
                        "Sequence name from a custom specified region not found in the input "
                        "sequence file! region: " +
                        secondary::region_to_string(region));
            }
            const int64_t seq_id = it->second;
            ret[seq_id].emplace_back(secondary::Region{region.name, region.start, region.end});
        }
    }

    // Divide draft sequences into groups of specified size, as sort of a barrier.
    std::vector<polisher::Interval> region_batches = create_batches(
            ret, draft_batch_size, [](const std::vector<secondary::Region>& regions) {
                int64_t sum = 0;
                for (const auto& region : regions) {
                    sum += region.end - region.start;
                }
                return sum;
            });

    return std::make_pair(std::move(ret), std::move(region_batches));
}

}  // namespace dorado::secondary
