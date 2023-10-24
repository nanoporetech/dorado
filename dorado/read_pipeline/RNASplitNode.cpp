#include "RNASplitNode.h"

#include "read_utils.h"
#include "utils/alignment_utils.h"
#include "utils/duplex_utils.h"
#include "utils/sequence_utils.h"
#include "utils/time_utils.h"
#include "utils/uuid_utils.h"

#include <spdlog/spdlog.h>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <optional>
#include <string>

namespace {

using namespace dorado;

typedef RNASplitNode::PosRange PosRange;
typedef RNASplitNode::PosRanges PosRanges;

template <class FilterF>
auto filter_ranges(const PosRanges& ranges, FilterF filter_f) {
    PosRanges filtered;
    std::copy_if(ranges.begin(), ranges.end(), std::back_inserter(filtered), filter_f);
    return filtered;
}

//merges overlapping ranges and ranges separated by merge_dist or less
//ranges supposed to be sorted by start coordinate
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

std::vector<std::pair<uint64_t, uint64_t>> detect_pore_signal(const torch::Tensor& signal,
                                                              float threshold,
                                                              uint64_t cluster_dist,
                                                              uint64_t ignore_prefix) {
    std::vector<std::pair<uint64_t, uint64_t>> ans;
    auto pore_a = signal.accessor<int16_t, 1>();
    int64_t cl_start = -1;
    int64_t cl_end = -1;

    for (auto i = ignore_prefix; i < pore_a.size(0); i++) {
        if (pore_a[i] > threshold) {
            //check if we need to start new cluster
            if (cl_end == -1 || i > cl_end + cluster_dist) {
                //report previous cluster
                if (cl_end != -1) {
                    assert(cl_start != -1);
                    ans.push_back({cl_start, cl_end});
                }
                cl_start = i;
            }
            cl_end = i + 1;
        }
    }
    //report last cluster
    if (cl_end != -1) {
        assert(cl_start != -1);
        assert(cl_start < pore_a.size(0) && cl_end <= pore_a.size(0));
        ans.push_back({cl_start, cl_end});
    }

    return ans;
}

//If read.parent_read_id is not empty then it will be used as parent_read_id of the subread
//signal_range should already be 'adjusted' to stride (e.g. probably gotten from seq_range)
SimplexReadPtr subread(const SimplexRead& read, PosRange signal_range) {
    //TODO support mods
    //NB: currently doesn't support mods
    const int stride = read.read_common.model_stride;
    assert(signal_range.first <= signal_range.second);
    assert(signal_range.first % stride == 0);
    assert(signal_range.second % stride == 0 ||
           (signal_range.second == read.read_common.get_raw_data_samples()));
    assert(read.read_common.seq.empty());
    assert(read.read_common.qstring.empty());

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

    assert(signal_range.second == read.read_common.get_raw_data_samples());

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

}  // namespace

namespace dorado {

RNASplitNode::ExtRead RNASplitNode::create_ext_read(SimplexReadPtr r) const {
    ExtRead ext_read;
    ext_read.read = std::move(r);
    ext_read.data_as_int16 = ext_read.read->read_common.raw_data;
    ext_read.possible_pore_regions = possible_pore_regions(ext_read);
    return ext_read;
}

PosRanges RNASplitNode::possible_pore_regions(const RNASplitNode::ExtRead& read) const {
    spdlog::trace("Analyzing signal in read {}", read.read->read_common.read_id);

    auto pore_sample_ranges =
            detect_pore_signal(read.data_as_int16, m_settings.pore_thr, m_settings.pore_cl_dist,
                               m_settings.expect_pore_prefix);

    PosRanges pore_regions;
    for (auto pore_sample_range : pore_sample_ranges) {
        auto start_pos = pore_sample_range.first;
        auto end_pos = pore_sample_range.second;
        pore_regions.push_back({start_pos, end_pos});
        spdlog::info("Pore range {}-{}", start_pos, end_pos);
    }

    return pore_regions;
}

std::vector<SimplexReadPtr> RNASplitNode::subreads(SimplexReadPtr read,
                                                   const std::vector<PosRange>& spacers) const {
    std::vector<SimplexReadPtr> subreads;
    subreads.reserve(spacers.size() + 1);

    if (spacers.empty()) {
        subreads.push_back(std::move(read));
        return subreads;
    }

    uint64_t start_pos = 0;
    for (auto r : spacers) {
        if (start_pos < r.first) {
            subreads.push_back(subread(*read, {start_pos, r.first}));
        }
        start_pos = r.second;
    }
    if (start_pos < read->read_common.get_raw_data_samples()) {
        subreads.push_back(subread(*read, {start_pos, read->read_common.get_raw_data_samples()}));
    }

    return subreads;
}

std::vector<std::pair<std::string, RNASplitNode::SplitFinderF>> RNASplitNode::build_split_finders()
        const {
    std::vector<std::pair<std::string, SplitFinderF>> split_finders;
    split_finders.push_back({"PORE_ADAPTER", [&](const ExtRead& read) {
                                 return filter_ranges(read.possible_pore_regions,
                                                      [&](PosRange r) { return true; });
                             }});

    return split_finders;
}

std::vector<SimplexReadPtr> RNASplitNode::split(SimplexReadPtr init_read) const {
    using namespace std::chrono;

    auto start_ts = high_resolution_clock::now();
    auto read_id = init_read->read_common.read_id;
    spdlog::trace("Processing read {}", read_id);

    std::vector<ExtRead> to_split;
    to_split.push_back(create_ext_read(std::move(init_read)));
    for (const auto& [description, split_f] : m_split_finders) {
        spdlog::trace("Running {}", description);
        std::vector<ExtRead> split_round_result;
        for (auto& r : to_split) {
            auto spacers = split_f(r);
            spdlog::trace("DSN: {} strategy {} splits in read {}", description, spacers.size(),
                          read_id);

            if (spacers.empty()) {
                split_round_result.push_back(std::move(r));
            } else {
                for (auto& sr : subreads(std::move(r.read), spacers)) {
                    split_round_result.push_back(create_ext_read(std::move(sr)));
                }
            }
        }
        to_split = std::move(split_round_result);
    }

    std::vector<SimplexReadPtr> split_result;
    size_t subread_id = 0;
    for (auto& ext_read : to_split) {
        if (!ext_read.read->read_common.parent_read_id.empty()) {
            ext_read.read->read_common.subread_id = subread_id++;
            ext_read.read->read_common.split_count = to_split.size();
            const auto subread_uuid =
                    utils::derive_uuid(ext_read.read->read_common.parent_read_id,
                                       std::to_string(ext_read.read->read_common.subread_id));
            ext_read.read->read_common.read_id = subread_uuid;
        }

        split_result.push_back(std::move(ext_read.read));
    }

    // Adjust prev and next read ids.
    if (split_result.size() > 1) {
        for (int i = 0; i < split_result.size(); i++) {
            if (i == split_result.size() - 1) {
                // For the last split read, the next read remains the same as the
                // original read's next read.
                split_result[i]->prev_read = split_result[i - 1]->read_common.read_id;
            } else if (i == 0) {
                // For the first split read, the previous read remains the same as the
                // original read's previous read.
                split_result[i]->next_read = split_result[i + 1]->read_common.read_id;
            } else {
                split_result[i]->prev_read = split_result[i - 1]->read_common.read_id;
                split_result[i]->next_read = split_result[i + 1]->read_common.read_id;
            }
        }
    }

    spdlog::trace("Read {} split into {} subreads", read_id, split_result.size());

    auto stop_ts = high_resolution_clock::now();
    spdlog::trace("READ duration: {} microseconds (ID: {})",
                  duration_cast<microseconds>(stop_ts - start_ts).count(), read_id);

    return split_result;
}

void RNASplitNode::worker_thread() {
    torch::InferenceMode inference_mode_guard;

    Message message;
    while (get_input_message(message)) {
        // If this message isn't a read, just forward it to the sink.
        if (!std::holds_alternative<SimplexReadPtr>(message)) {
            send_message_to_sink(std::move(message));
            continue;
        }

        // If this message isn't a read, we'll get a bad_variant_access exception.
        auto init_read = std::get<SimplexReadPtr>(std::move(message));
        spdlog::info("About to split");
        for (auto& subread : split(std::move(init_read))) {
            //TODO correctly process end_reason when we have them
            send_message_to_sink(std::move(subread));
        }
    }
}

RNASplitNode::RNASplitNode(RNASplitSettings settings, int num_worker_threads, size_t max_reads)
        : MessageSink(max_reads),
          m_settings(std::move(settings)),
          m_num_worker_threads(num_worker_threads) {
    m_split_finders = build_split_finders();
    start_threads();
    spdlog::info("Create rna splitter");
}

void RNASplitNode::start_threads() {
    for (int i = 0; i < m_num_worker_threads; ++i) {
        m_worker_threads.push_back(
                std::make_unique<std::thread>(&RNASplitNode::worker_thread, this));
    }
}

void RNASplitNode::terminate_impl() {
    terminate_input_queue();

    // Wait for all the Node's worker threads to terminate
    for (auto& t : m_worker_threads) {
        if (t->joinable()) {
            t->join();
        }
    }
    m_worker_threads.clear();
}

void RNASplitNode::restart() {
    restart_input_queue();
    start_threads();
}

stats::NamedStats RNASplitNode::sample_stats() const { return stats::from_obj(m_work_queue); }

}  // namespace dorado
