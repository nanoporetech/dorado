#include "splitter_utils.h"

#include "read_utils.h"
#include "utils/time_utils.h"

#include <optional>

namespace dorado::splitter {

SimplexReadPtr subread(const SimplexRead& read,
                       std::optional<PosRange> seq_range,
                       PosRange signal_range) {
    //TODO support mods
    //NB: currently doesn't support mods
    //assert(read.mod_base_info == nullptr && read.base_mod_probs.empty());
    if (read.read_common.mod_base_info != nullptr || !read.read_common.base_mod_probs.empty()) {
        throw std::runtime_error(std::string("Read splitting doesn't support mods yet"));
    }

    auto subread = utils::shallow_copy_read(read);

    subread->read_common.read_tag = read.read_common.read_tag;
    subread->read_common.client_id = read.read_common.client_id;
    subread->read_common.raw_data = subread->read_common.raw_data.index(
            {torch::indexing::Slice(signal_range.first, signal_range.second)});
    subread->read_common.attributes.read_number = -1;

    //we adjust for it in new start time
    subread->read_common.attributes.num_samples = signal_range.second - signal_range.first;
    subread->read_common.num_trimmed_samples = 0;
    subread->start_sample =
            read.start_sample + read.read_common.num_trimmed_samples + signal_range.first;
    subread->end_sample = subread->start_sample + subread->read_common.attributes.num_samples;

    auto start_time_ms = read.run_acquisition_start_time_ms +
                         static_cast<uint64_t>(std::round(subread->start_sample * 1000. /
                                                          subread->read_common.sample_rate));
    subread->read_common.attributes.start_time =
            utils::get_string_timestamp_from_unix_time(start_time_ms);
    subread->read_common.start_time_ms = start_time_ms;

    if (seq_range) {
        const int stride = read.read_common.model_stride;
        assert(signal_range.first <= signal_range.second);
        assert(signal_range.first / stride <= read.read_common.moves.size());
        assert(signal_range.second / stride <= read.read_common.moves.size());
        assert(signal_range.first % stride == 0);
        assert(signal_range.second % stride == 0 ||
               (signal_range.second == read.read_common.get_raw_data_samples() &&
                seq_range->second == read.read_common.seq.size()));

        subread->read_common.seq = subread->read_common.seq.substr(
                seq_range->first, seq_range->second - seq_range->first);
        subread->read_common.qstring = subread->read_common.qstring.substr(
                seq_range->first, seq_range->second - seq_range->first);
        subread->read_common.moves = std::vector<uint8_t>(
                subread->read_common.moves.begin() + signal_range.first / stride,
                subread->read_common.moves.begin() + signal_range.second / stride);
        assert(signal_range.second == read.read_common.get_raw_data_samples() ||
               subread->read_common.moves.size() * stride ==
                       subread->read_common.get_raw_data_samples());
    }

    // Initialize the subreads previous and next reads with the parent's ids.
    // These are updated at the end when all subreads are available.
    subread->prev_read = read.prev_read;
    subread->next_read = read.next_read;

    if (!read.read_common.parent_read_id.empty()) {
        subread->read_common.parent_read_id = read.read_common.parent_read_id;
    } else {
        subread->read_common.parent_read_id = read.read_common.read_id;
    }
    return subread;
}

PosRanges merge_ranges(const PosRanges& ranges, uint64_t merge_dist) {
    PosRanges merged;
    for (auto& r : ranges) {
        assert(merged.empty() || r.first >= merged.back().first);
        if (merged.empty() || r.first > merged.back().second + merge_dist) {
            merged.push_back(r);
        } else {
            merged.back().second = r.second;
        }
    }
    return merged;
}

}  // namespace dorado::splitter
