#include "DuplexSplitNode.h"

#include "utils/sequence_utils.h"
#include "utils/duplex_utils.h"
#include "3rdparty/edlib/edlib/include/edlib.h"

#include <spdlog/spdlog.h>
#include <optional>

namespace {

using namespace dorado;

typedef DuplexSplitNode::PosRange PosRange;

std::ostream& operator<<(std::ostream& os, const PosRange& r)
{
    return os << "[" << r.first << ", " << r.second << "]";
}

//                           T  A     T        T  C     A     G        T     A  C
//std::vector<uint8_t> moves{1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0};
std::vector<uint64_t> move_cum_sums(const std::vector<uint8_t> moves) {
    std::vector<uint64_t> ans(moves.size(), 0);
    if (!moves.empty()) {
        ans[0] = moves[0];
    }
    for (size_t i = 1, n = moves.size(); i < n; i++) {
        ans[i] = ans[i-1] + moves[i];
    }
    return ans;
}

std::vector<std::pair<size_t, size_t>> detect_pore_signal(torch::Tensor pa_signal,
                                                          float threshold,
                                                          size_t cluster_dist) {
    spdlog::info("Max raw signal {} pA", pa_signal.max().item<float>());

    std::vector<std::pair<size_t, size_t>> ans;
    //FIXME what type to use here?
    auto pore_a = pa_signal.accessor<float, 1>();
    size_t start = size_t(-1);
    size_t end = 0;

    for (size_t i = 0; i < pore_a.size(0); i++) {
        if (pore_a[i] > threshold) {
            if (end == 0 || i > end + cluster_dist) {
                if (end > 0) {
                    ans.push_back({start, end});
                }
                start = i;
                end = i + 1;
            }
            end = i + 1;
        }
    }
    if (end > 0) {
        ans.push_back(std::pair{start, end});
    }
    return ans;
    //std::vector<std::pair<size_t, size_t>> ans;
    // auto pore_positions = torch::nonzero(pa_signal > threshold);
    // //FIXME what type to use here?
    // auto pore_a = pore_positions.accessor<int32_t, 1>();
    // size_t start = size_t(-1);
    // size_t end = size_t(-1);

    // int i = 0;
    // for (; i < pore_a.size(0); i++) {
    //     if (i == 0 || pore_a[i] > end + cluster_dist) {
    //         if (i > 0) {
    //             ans.push_back(std::pair{start, end});
    //         }
    //         start = pore_a[i];
    //     }
    //     end = pore_a[i] + 1;
    // }
    // if (i > 0) {
    //     ans.push_back(std::pair{start, end});
    // }
    // return ans;
}

//[inc, excl)
std::optional<PosRange>
find_best_adapter_match(const std::string& adapter,
                        const std::string& seq,
                        int dist_thr,
                        std::optional<PosRange> subrange = std::nullopt) {
    uint64_t shift = 0;
    uint64_t span = seq.size();
    if (subrange) {
        assert(subrange->first <= subrange->second && subrange->second <= seq.size());
        shift = subrange->first;
        span = subrange->second - subrange->first;
    }
    //might be unnecessary, depending on edlib's empty sequence handling
    if (span == 0) return std::nullopt;

    auto edlib_result = edlibAlign(adapter.c_str(), adapter.size(),
                             seq.c_str() + shift, span,
                             edlibNewAlignConfig(-1, EDLIB_MODE_HW, EDLIB_TASK_LOC, NULL, 0));
    assert(edlib_result.status == EDLIB_STATUS_OK);
    std::optional<PosRange> res = std::nullopt;
    if (edlib_result.status == EDLIB_STATUS_OK && edlib_result.editDistance != -1) {
        //spdlog::info("ed {}", edlib_result.editDistance);
        //spdlog::info("al {}", edlib_result.alignmentLength);
        //spdlog::info("sl {}", edlib_result.startLocations[0]);
        //spdlog::info("el {}", edlib_result.endLocations[0]);

        //FIXME REMOVE and use dist_thr instead of -1
        //only report for top call on the read
        if (span == seq.size() - 200 /*expect_adapter_prefix*/) {
            //FIXME remove
            spdlog::info("Best adapter match edit distance: {}", edlib_result.editDistance);
        }
        if (edlib_result.editDistance <= dist_thr) {
            res = {edlib_result.startLocations[0] + shift, edlib_result.endLocations[0] + shift + 1};
        }
    }
    edlibFreeAlignResult(edlib_result);
    return res;
}

std::vector<PosRange> find_adapter_matches(const std::string& adapter,
                                           const std::string& seq,
                                           int dist_thr,
                                           uint64_t ignore_prefix) {
    std::vector<PosRange> answer;
    if (seq.size() <= ignore_prefix) return answer;

    if (auto best_match = find_best_adapter_match(adapter, seq, dist_thr,
                                        PosRange{ignore_prefix, seq.size()})) {
        //Try to split again each side
        if (auto left_match = find_best_adapter_match(adapter, seq, dist_thr,
                                        PosRange{ignore_prefix, best_match->first})) {
            answer.push_back(*left_match);
        }
        answer.push_back(*best_match);
        if (auto right_match = find_best_adapter_match(adapter, seq, dist_thr,
                                        PosRange{best_match->second, seq.size()})) {
            answer.push_back(*right_match);
        }
    }
    return answer;
}

//first -- pore regions matched with adapters
//second -- remaining pore regions and 'empty' ranges before unmatched adapters
std::pair<std::vector<PosRange>, std::vector<PosRange>>
match_pore_adapter(const std::vector<PosRange>& pore_regions,
                   const std::vector<PosRange>& adapter_regions,
                   uint64_t max_dist) {
    std::vector<PosRange> matched_pore_regions;
    std::vector<PosRange> extra_potential_splits;
    const size_t ar_size = adapter_regions.size();
    size_t ar_idx = 0;
    uint64_t ar_start;
    for (auto pore_region : pore_regions) {
        bool found_match = false;
        while (ar_idx < ar_size
            && (ar_start = adapter_regions[ar_idx].first) < pore_region.first) {
            extra_potential_splits.push_back({ar_start, ar_start});
            ar_idx++;
        }
        assert(ar_idx == ar_size || ar_start == adapter_regions[ar_idx].first);
        if (ar_idx < ar_size && ar_start < pore_region.second + max_dist) {
            //match!
            matched_pore_regions.push_back(pore_region);
            found_match = true;
            ar_idx++;
        }
        if (!found_match) {
            extra_potential_splits.push_back(pore_region);
        }
    }
    for (; ar_idx < ar_size; ar_idx++) {
        ar_start = adapter_regions[ar_idx].first;
        extra_potential_splits.push_back({ar_start, ar_start});
    }
    return {matched_pore_regions, extra_potential_splits};
}

//we probably don't need merging actually, just assert if overlap for now
std::vector<PosRange>
merge_ranges(std::vector<PosRange>&& pore_regions) {
    for (size_t i = 1; i < pore_regions.size(); i++) {
        assert(pore_regions[i-1].second < pore_regions[i].first);
    }
    return pore_regions;
}

bool check_rc_match(const std::string& seq, PosRange templ_r, PosRange compl_r, int dist_thr) {
    const char* c_seq = seq.c_str();
    std::vector<char> rc_compl(c_seq + compl_r.first, c_seq + compl_r.second);
    dorado::utils::reverse_complement(rc_compl);

    auto edlib_result = edlibAlign(c_seq + templ_r.first,
                            templ_r.second - templ_r.first,
                            rc_compl.data(), rc_compl.size(),
                            edlibNewAlignConfig(dist_thr, EDLIB_MODE_SHW, EDLIB_TASK_DISTANCE, NULL, 0));
    assert(edlib_result.status == EDLIB_STATUS_OK);
    std::optional<PosRange> res = std::nullopt;
    assert(edlib_result.status == EDLIB_STATUS_OK && edlib_result.editDistance <= dist_thr);
    spdlog::debug("Checking ranges [{}, {}] vs [{}, {}]: edist={}",
                    templ_r.first, templ_r.second,
                    compl_r.first, compl_r.second,
                    edlib_result.editDistance);
    bool match = (edlib_result.status == EDLIB_STATUS_OK) && (edlib_result.editDistance != -1);
    edlibFreeAlignResult(edlib_result);
    return match;
}

std::shared_ptr<Read> subread(const Read& read, PosRange seq_range, PosRange signal_range) {
    //FIXME TODO
    //FIXME do we have end_reason access?

    /*
    new_read->raw_data = ...read->raw_data.index({torch::indexing::Slice(trim_start, torch::indexing::None)});
    new_read->sample_rate = run_sample_rate;
    auto start_time_ms =
            run_acquisition_start_time_ms + ((read_data.start_sample * 1000) / run_sample_rate);
    auto start_time = get_string_timestamp_from_unix_time(start_time_ms);
    new_read->scaling = read_data.calibration_scale;
    new_read->offset = read_data.calibration_offset;
    new_read->read_id = std::move(read_id_str);
    new_read->num_trimmed_samples = 0;
    new_read->attributes.read_number = read_data.read_number;
    new_read->attributes.fast5_filename = std::filesystem::path(path.c_str()).filename().string();
    new_read->attributes.mux = read_data.well;
    new_read->attributes.channel_number = read_data.channel;
    new_read->attributes.start_time = start_time;
    new_read->run_id = run_info_data->protocol_run_id;
    new_read->scale = ...
    new_read->shift = ...
    new_read->seq = ...
    new_read->qstring = ...
    new_read->moves = ...
    new_read->model_stride = ...
    */
    return std::make_shared<Read>();
}

}

namespace dorado {

std::vector<DuplexSplitNode::PosRange>
DuplexSplitNode::possible_pore_regions(const Read& read) {
    std::vector<DuplexSplitNode::PosRange> pore_regions;

    const auto move_sums = move_cum_sums(read.moves);
    assert(move_sums.back() == read.seq.length());

    //pA formula before scaling:
    //pA = read->scaling * (raw + read->offset);
    //pA formula after scaling:
    //pA = read->scale * raw + read->shift
    for (auto pore_signal_region : detect_pore_signal(
                                   read.raw_data.to(torch::kFloat) * read.scale + read.shift,
                                   m_settings.pore_thr,
                                   m_settings.pore_cl_dist)) {
        auto move_start = pore_signal_region.first / read.model_stride;
        auto move_end = pore_signal_region.second / read.model_stride;
        assert(move_end >= move_start);
        if (move_sums.at(move_start) == 0) {
            //basecalls have not started yet
            continue;
        }
        auto start_pos = move_sums.at(move_start) - 1;
        //NB. adding adapter length
        auto end_pos = move_sums.at(move_end);
        assert(end_pos > start_pos);
        pore_regions.push_back({start_pos, end_pos});
    }
    return pore_regions;
}

//empty return means that there is no need to split
//std::vector<ReadRange>
std::vector<DuplexSplitNode::PosRange>
DuplexSplitNode::identify_splits(const Read& read) {
    spdlog::info("DSN: Searching for splits in read {}", read.read_id);
    std::vector<PosRange> interspace_regions;

    auto pore_region_candidates = possible_pore_regions(read);
    auto adapter_region_candidates = find_adapter_matches(m_settings.adapter, read.seq,
                                            m_settings.adapter_edist,
                                            m_settings.expect_adapter_prefix);

    auto [definite_splits, extra_regions] =
                match_pore_adapter(pore_region_candidates,
                                   adapter_region_candidates,
                                   m_settings.pore_adapter_gap_thr);
    auto matched_pore_adapter = definite_splits.size();

    //checking reverse-complement matching for uncertain regions
    size_t templ_flank = m_settings.flank_size;
    //No need to 'cut' adapter region -- matching complement region
    // won't be penalized for having extra prefix in edlib
    size_t compl_flank = m_settings.flank_size + m_settings.adapter.size();
    for (auto r : extra_regions) {
        if (r.first < templ_flank ||
            r.second + compl_flank > read.seq.length()) {
            //TODO maybe handle some of these cases too (extra min_avail_sequence parameter?)
            continue;
        }
        //FIXME any need to adjust for adapter
        //FIXME should we subtract a portion of tail adapter from the first region too?
        if (check_rc_match(read.seq,
                        {r.first - templ_flank, r.first},
                        {r.second, r.second + compl_flank},
                        m_settings.flank_edist)) {
            //TODO might make sense to use actual adapter coordinates when it was found
            definite_splits.push_back(r);
        }
    }
    spdlog::info("DSN: Read {} ; pore regions {} ; adapter regions {} ; matched {} ; rc checked {} ; rc check pass {} ; final splits {}",
                    read.read_id, pore_region_candidates.size(),
                    adapter_region_candidates.size(), matched_pore_adapter,
                    extra_regions.size(), definite_splits.size() - matched_pore_adapter,
                    definite_splits.size());
    std::sort(definite_splits.begin(), definite_splits.end());
    return merge_ranges(std::move(definite_splits));
}

std::vector<Message> DuplexSplitNode::split(const Read& read,
                                            const std::vector<PosRange> &interspace_regions) {
    assert(!interspace_regions.empty());
    std::vector<Message> ans;
    ans.reserve(interspace_regions.size() + 1);

    const auto seq_to_sig_map = utils::moves_to_map(
            read.moves, read.model_stride, read.raw_data.size(0), read.seq.size() + 1);

    //TODO maybe simplify by adding begin/end stubs?
    uint64_t start_pos = 0;
    uint64_t signal_start = seq_to_sig_map[0];
    for (auto& r: interspace_regions) {
        //FIXME Don't forget to correctly process end_reasons!
        ans.push_back(subread(read, {start_pos, r.first}, {signal_start, seq_to_sig_map[r.first]}));
        start_pos = r.second;
        signal_start = seq_to_sig_map[r.second];
    }
    ans.push_back(subread(read, {start_pos, read.seq.size()}, {signal_start, read.raw_data.size(0)}));
    return ans;
}

void DuplexSplitNode::worker_thread() {
    spdlog::info("DSN: Hello from worker thread");
    Message message;
    while (m_work_queue.try_pop(message)) {
        // If this message isn't a read, we'll get a bad_variant_access exception.
        auto read = std::get<std::shared_ptr<Read>>(message);

        auto ranges = identify_splits(*read);
        std::ostringstream oss;
        std::copy(ranges.begin(), ranges.end(), std::ostream_iterator<PosRange>(oss, ";"));
        //spdlog::info("DSN: identified {} splits in read {}: {}", ranges.size(), read->read_id, oss.str());
        //if (ranges.empty()) {
        //    m_sink.push_message(read);
        //} else {
        //    for (auto m : split(*read, ranges)) {
        //        m_sink.push_message(std::move(m));
        //    }
        //}
    }
}

DuplexSplitNode::DuplexSplitNode(MessageSink& sink, DuplexSplitSettings settings,
                                int num_worker_threads, size_t max_reads)
        : MessageSink(max_reads), m_sink(sink),
            m_settings(settings),
            m_num_worker_threads(num_worker_threads) {
    for (int i = 0; i < m_num_worker_threads; i++) {
        std::unique_ptr<std::thread> split_worker_thread =
                std::make_unique<std::thread>(&DuplexSplitNode::worker_thread, this);
        worker_threads.push_back(std::move(split_worker_thread));
        //worker_threads.push_back(std::make_unique<std::thread>(&DuplexSplitNode::worker_thread, this));
    }
}

DuplexSplitNode::~DuplexSplitNode() {
    terminate();

    // Wait for all the Node's worker threads to terminate
    for (auto& t : worker_threads) {
        t->join();
    }

    // Notify the sink that the Node has terminated
    m_sink.terminate();
}

}