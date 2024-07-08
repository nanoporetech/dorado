#include "splitter_utils.h"

#include "read_pipeline/ReadPipeline.h"
#include "read_pipeline/read_utils.h"
#include "utils/time_utils.h"

#include <ATen/TensorIndexing.h>

namespace dorado::splitter {
namespace {
// This part of subread() is split out into its own unoptimised function since not doing so
// causes binaries built by GCC8 with ASAN enabled to crash during static init.
// Note that the cause of the crash doesn't appear to be specific to this bit of code, since
// removing other parts from subread() also "fixes" the issue, but this is the smallest
// snippet that works around the issue without potentially incurring performance issues.
#if defined(__GNUC__) && defined(__SANITIZE_ADDRESS__)
__attribute__((optimize("O0")))
#endif
 void assign_subread_parent_id(const SimplexRead& read, SimplexReadPtr & subread) {
    if (!read.read_common.parent_read_id.empty()) {
        subread->read_common.parent_read_id = read.read_common.parent_read_id;
    } else {
        subread->read_common.parent_read_id = read.read_common.read_id;
    }
}
}  // namespace

SimplexReadPtr subread(const SimplexRead& read,
                       std::optional<PosRange> seq_range,
                       std::pair<uint64_t, uint64_t> signal_range) {
    //TODO support mods
    //NB: currently doesn't support mods
    //assert(read.mod_base_info == nullptr && read.base_mod_probs.empty());
    if (read.read_common.mod_base_info != nullptr || !read.read_common.base_mod_probs.empty()) {
        throw std::runtime_error(std::string("Read splitting doesn't support mods yet"));
    }

    auto subread = utils::shallow_copy_read(read);

    subread->read_common.raw_data = subread->read_common.raw_data.index(
            {at::indexing::Slice(signal_range.first, signal_range.second)});
    subread->read_common.attributes.read_number = -1;

    //we adjust for it in new start time
    subread->read_common.split_point = uint32_t(signal_range.first);
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
        subread->read_common.pre_trim_seq_length = subread->read_common.seq.length();
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

    assign_subread_parent_id(read, subread);
    return subread;
}

PosRanges merge_ranges(const PosRanges& ranges, uint64_t merge_dist) {
    PosRanges merged;
    for (auto& r : ranges) {
        assert(merged.empty() || r.first >= merged.back().first);
        if (merged.empty() || r.first > merged.back().second + merge_dist) {
            merged.push_back(r);
        } else {
            merged.back().second = std::max(r.second, merged.back().second);
        }
    }
    return merged;
}

}  // namespace dorado::splitter
