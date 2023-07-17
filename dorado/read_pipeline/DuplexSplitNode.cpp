#include "DuplexSplitNode.h"

#include "utils/alignment_utils.h"
#include "utils/duplex_utils.h"
#include "utils/read_utils.h"
#include "utils/sequence_utils.h"
#include "utils/time_utils.h"
#include "utils/uuid_utils.h"

#include <openssl/sha.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <optional>
#include <string>

namespace {

using namespace dorado;

typedef DuplexSplitNode::PosRange PosRange;
typedef DuplexSplitNode::PosRanges PosRanges;

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
    auto pore_a = signal.accessor<float, 1>();
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
        ans.push_back(std::pair{cl_start, cl_end});
    }

    return ans;
}

//[start, end)
std::optional<PosRange> find_best_adapter_match(const std::string& adapter,
                                                const std::string& seq,
                                                int dist_thr,
                                                PosRange subrange) {
    assert(subrange.first <= subrange.second && subrange.second <= seq.size());
    auto shift = subrange.first;
    auto span = subrange.second - subrange.first;

    if (span == 0)
        return std::nullopt;

    auto edlib_cfg = edlibNewAlignConfig(dist_thr, EDLIB_MODE_HW, EDLIB_TASK_LOC, NULL, 0);

    auto edlib_result =
            edlibAlign(adapter.c_str(), adapter.size(), seq.c_str() + shift, span, edlib_cfg);
    assert(edlib_result.status == EDLIB_STATUS_OK);
    std::optional<PosRange> res = std::nullopt;
    if (edlib_result.status == EDLIB_STATUS_OK && edlib_result.editDistance != -1) {
        assert(edlib_result.editDistance <= dist_thr);
        res = {edlib_result.startLocations[0] + shift, edlib_result.endLocations[0] + shift + 1};
    }
    edlibFreeAlignResult(edlib_result);
    return res;
}

//currently just finds a single best match
//TODO efficiently find more matches
std::vector<PosRange> find_adapter_matches(const std::string& adapter,
                                           const std::string& seq,
                                           int dist_thr,
                                           uint64_t ignore_prefix) {
    std::vector<PosRange> answer;
    if (ignore_prefix < seq.size()) {
        if (auto best_match =
                    find_best_adapter_match(adapter, seq, dist_thr, {ignore_prefix, seq.size()})) {
            answer.push_back(*best_match);
        }
    }
    return answer;
}

//semi-global alignment of "template region" to "complement region"
//returns range in the compl_r
std::optional<PosRange> check_rc_match(const std::string& seq,
                                       PosRange templ_r,
                                       PosRange compl_r,
                                       int dist_thr) {
    assert(templ_r.second > templ_r.first);
    assert(compl_r.second > compl_r.first);
    assert(dist_thr >= 0);

    auto rc_compl = dorado::utils::reverse_complement(
            seq.substr(compl_r.first, compl_r.second - compl_r.first));

    auto edlib_cfg = edlibNewAlignConfig(dist_thr, EDLIB_MODE_HW, EDLIB_TASK_LOC, NULL, 0);

    auto edlib_result = edlibAlign(seq.c_str() + templ_r.first, templ_r.second - templ_r.first,
                                   rc_compl.c_str(), rc_compl.size(), edlib_cfg);
    assert(edlib_result.status == EDLIB_STATUS_OK);

    bool match = (edlib_result.status == EDLIB_STATUS_OK) && (edlib_result.editDistance != -1);
    std::optional<PosRange> res = std::nullopt;
    if (match) {
        assert(edlib_result.editDistance <= dist_thr);
        assert(edlib_result.numLocations > 0 && edlib_result.endLocations[0] < compl_r.second &&
               edlib_result.startLocations[0] < compl_r.second);
        res = PosRange(compl_r.second - edlib_result.endLocations[0],
                       compl_r.second - edlib_result.startLocations[0]);
    }

    edlibFreeAlignResult(edlib_result);
    return res;
}

//TODO end_reason access?
//If read.parent_read_id is not empty then it will be used as parent_read_id of the subread
//signal_range should already be 'adjusted' to stride (e.g. probably gotten from seq_range)
std::shared_ptr<Read> subread(const Read& read, PosRange seq_range, PosRange signal_range) {
    //TODO support mods
    //NB: currently doesn't support mods
    //assert(read.base_mod_info == nullptr && read.base_mod_probs.empty());
    if (read.base_mod_info != nullptr || !read.base_mod_probs.empty()) {
        throw std::runtime_error(std::string("Read splitting doesn't support mods yet"));
    }
    const int stride = read.model_stride;
    assert(signal_range.first <= signal_range.second);
    assert(signal_range.first / stride <= read.moves.size());
    assert(signal_range.second / stride <= read.moves.size());
    assert(signal_range.first % stride == 0);
    assert(signal_range.second % stride == 0 ||
           (signal_range.second == read.raw_data.size(0) && seq_range.second == read.seq.size()));

    auto subread = utils::shallow_copy_read(read);

    const auto subread_id = utils::derive_uuid(
            read.read_id, std::to_string(seq_range.first) + "-" + std::to_string(seq_range.second));
    subread->read_id = subread_id;
    subread->read_tag = read.read_tag;
    subread->raw_data = subread->raw_data.index(
            {torch::indexing::Slice(signal_range.first, signal_range.second)});
    subread->attributes.read_number = -1;

    //we adjust for it in new start time
    subread->attributes.num_samples = signal_range.second - signal_range.first;
    subread->num_trimmed_samples = 0;
    subread->start_sample = read.start_sample + read.num_trimmed_samples + signal_range.first;
    subread->end_sample = subread->start_sample + subread->attributes.num_samples;

    auto start_time_ms = read.run_acquisition_start_time_ms +
                         uint64_t(std::round(subread->start_sample * 1000. / subread->sample_rate));
    subread->attributes.start_time = utils::get_string_timestamp_from_unix_time(start_time_ms);
    subread->start_time_ms = start_time_ms;

    subread->seq = subread->seq.substr(seq_range.first, seq_range.second - seq_range.first);
    subread->qstring = subread->qstring.substr(seq_range.first, seq_range.second - seq_range.first);
    subread->moves = std::vector<uint8_t>(subread->moves.begin() + signal_range.first / stride,
                                          subread->moves.begin() + signal_range.second / stride);
    assert(signal_range.second == read.raw_data.size(0) ||
           subread->moves.size() * stride == subread->raw_data.size(0));

    if (!read.parent_read_id.empty()) {
        subread->parent_read_id = read.parent_read_id;
    } else {
        subread->parent_read_id = read.read_id;
    }
    return subread;
}

}  // namespace

namespace dorado {

DuplexSplitNode::ExtRead DuplexSplitNode::create_ext_read(std::shared_ptr<Read> r) const {
    ExtRead ext_read;
    ext_read.read = r;
    ext_read.move_sums = utils::move_cum_sums(r->moves);
    assert(!ext_read.move_sums.empty());
    assert(ext_read.move_sums.back() == r->seq.length());
    ext_read.data_as_float32 = r->raw_data.to(torch::kFloat);
    ext_read.possible_pore_regions = possible_pore_regions(ext_read);
    return ext_read;
}

PosRanges DuplexSplitNode::possible_pore_regions(const DuplexSplitNode::ExtRead& read) const {
    spdlog::trace("Analyzing signal in read {}", read.read->read_id);

    auto pore_sample_ranges =
            detect_pore_signal(read.data_as_float32, m_settings.pore_thr, m_settings.pore_cl_dist,
                               m_settings.expect_pore_prefix);

    PosRanges pore_regions;
    for (auto pore_sample_range : pore_sample_ranges) {
        auto move_start = pore_sample_range.first / read.read->model_stride;
        auto move_end = pore_sample_range.second / read.read->model_stride;
        assert(move_end >= move_start);
        //NB move_start can get to move_sums.size(), because of the stride rounding?
        if (move_start >= read.move_sums.size() || move_end >= read.move_sums.size() ||
            read.move_sums[move_start] == 0) {
            //either at very end of the signal or basecalls have not started yet
            continue;
        }
        auto start_pos = read.move_sums[move_start] - 1;
        //NB. adding adapter length
        auto end_pos = read.move_sums[move_end];
        assert(end_pos > start_pos);
        if (end_pos <= start_pos + m_settings.max_pore_region) {
            pore_regions.push_back({start_pos, end_pos});
        }
    }

    return pore_regions;
}

bool DuplexSplitNode::check_nearby_adapter(const Read& read, PosRange r, int adapter_edist) const {
    return find_best_adapter_match(m_settings.adapter, read.seq, adapter_edist,
                                   //including spacer region in search
                                   {r.first, std::min(r.second + m_settings.pore_adapter_range,
                                                      (uint64_t)read.seq.size())})
            .has_value();
}

//'spacer' is region potentially containing templ/compl strand boundary
//returns optional pair of matching ranges (first strictly to the left of spacer region)
std::optional<std::pair<PosRange, PosRange>>
DuplexSplitNode::check_flank_match(const Read& read, PosRange spacer, float err_thr) const {
    const uint64_t rlen = read.seq.length();
    assert(spacer.first <= spacer.second && spacer.second <= rlen);
    if (spacer.first <= m_settings.strand_end_trim || spacer.second == rlen) {
        return std::nullopt;
    }

    const uint64_t left_start = (spacer.first > m_settings.strand_end_flank)
                                        ? spacer.first - m_settings.strand_end_flank
                                        : 0;
    const uint64_t left_end = spacer.first - m_settings.strand_end_trim;
    assert(left_start < left_end);
    const uint64_t left_span = left_end - left_start;

    //including spacer region in search
    const uint64_t right_start = spacer.first;
    //(r.second - r.first) adjusts for potentially incorrectly detected split region
    //, shifting into correct sequence
    const uint64_t right_end = std::min(
            spacer.second + m_settings.strand_start_flank + (spacer.second - spacer.first), rlen);
    assert(right_start < right_end);
    const uint64_t right_span = right_end - right_start;

    const int dist_thr = std::round(err_thr * left_span);
    if (left_span >= m_settings.min_flank && right_span >= left_span) {
        if (auto match = check_rc_match(read.seq, {left_start, left_end},
                                        //including spacer region in search
                                        {right_start, right_end}, dist_thr)) {
            return std::pair{PosRange{left_start, left_end}, *match};
        }
    }
    return std::nullopt;
}

std::optional<DuplexSplitNode::PosRange> DuplexSplitNode::identify_middle_adapter_split(
        const Read& read) const {
    assert(m_settings.strand_end_flank > m_settings.strand_end_trim + m_settings.min_flank);
    const uint64_t r_l = read.seq.size();
    const uint64_t search_span =
            std::max(m_settings.middle_adapter_search_span,
                     uint64_t(std::round(m_settings.middle_adapter_search_frac * r_l)));
    if (r_l < search_span) {
        return std::nullopt;
    }

    spdlog::trace("Searching for adapter match");
    if (auto adapter_match = find_best_adapter_match(
                m_settings.adapter, read.seq, m_settings.relaxed_adapter_edist,
                {r_l / 2 - search_span / 2, r_l / 2 + search_span / 2})) {
        const uint64_t adapter_start = adapter_match->first;
        const uint64_t adapter_end = adapter_match->second;
        spdlog::trace("Checking middle match & start/end match");
        //Checking match around adapter
        if (check_flank_match(read, {adapter_start, adapter_start}, m_settings.flank_err)) {
            //Checking start/end match
            //some initializations might 'overflow' and not make sense, but not if check_rc_match below actually ends up checked!
            const uint64_t query_start = r_l - m_settings.strand_end_flank;
            const uint64_t query_end = r_l - m_settings.strand_end_trim;
            const uint64_t query_span = query_end - query_start;
            const int dist_thr = std::round(m_settings.flank_err * query_span);

            const uint64_t template_start = 0;
            const uint64_t template_end = std::min(m_settings.strand_start_flank, adapter_start);
            const uint64_t template_span = template_end - template_start;

            if (adapter_end + m_settings.strand_end_flank > r_l || template_span < query_span ||
                check_rc_match(
                        read.seq,
                        {r_l - m_settings.strand_end_flank, r_l - m_settings.strand_end_trim},
                        {0, std::min(m_settings.strand_start_flank, r_l)}, dist_thr)) {
                return PosRange{adapter_start - 1, adapter_start};
            }
        }
    }
    return std::nullopt;
}

std::optional<DuplexSplitNode::PosRange> DuplexSplitNode::identify_extra_middle_split(
        const Read& read) const {
    const uint64_t r_l = read.seq.size();
    //TODO parameterize
    const float ext_start_frac = 0.1;
    //extend to tolerate some extra length difference
    const uint64_t ext_start_flank =
            std::max(uint64_t(ext_start_frac * r_l), m_settings.strand_start_flank);
    //further consider only reasonably long reads
    if (ext_start_flank + m_settings.strand_end_flank > r_l) {
        return std::nullopt;
    }

    int flank_edist = std::round(m_settings.flank_err *
                                 (m_settings.strand_end_flank - m_settings.strand_end_trim));

    spdlog::trace("Checking start/end match");
    if (auto templ_start_match = check_rc_match(
                read.seq, {r_l - m_settings.strand_end_flank, r_l - m_settings.strand_end_trim},
                {0, std::min(r_l, ext_start_flank)}, flank_edist)) {
        //check if matched region and query overlap
        if (templ_start_match->second + m_settings.strand_end_flank > r_l) {
            return std::nullopt;
        }
        uint64_t est_middle = (templ_start_match->second + (r_l - m_settings.strand_end_flank)) / 2;
        spdlog::trace("Middle estimate {}", est_middle);
        //TODO parameterize
        const int min_split_margin = 100;
        const float split_margin_frac = 0.05;
        const auto split_margin = std::max(min_split_margin, int(split_margin_frac * r_l));

        spdlog::trace("Checking approx middle match");
        if (auto middle_match_ranges =
                    check_flank_match(read, {est_middle - split_margin, est_middle + split_margin},
                                      m_settings.flank_err)) {
            est_middle =
                    (middle_match_ranges->first.second + middle_match_ranges->second.first) / 2;
            spdlog::trace("Middle re-estimate {}", est_middle);
            return PosRange{est_middle - 1, est_middle};
        }
    }
    return std::nullopt;
}

std::vector<std::shared_ptr<Read>> DuplexSplitNode::subreads(
        std::shared_ptr<Read> read,
        const std::vector<PosRange>& spacers) const {
    if (spacers.empty()) {
        return {read};
    }

    std::vector<std::shared_ptr<Read>> subreads;
    subreads.reserve(spacers.size() + 1);

    const auto stride = read->model_stride;
    const auto seq_to_sig_map =
            utils::moves_to_map(read->moves, stride, read->raw_data.size(0), read->seq.size() + 1);

    //TODO maybe simplify by adding begin/end stubs?
    uint64_t start_pos = 0;
    uint64_t signal_start = seq_to_sig_map[0];
    for (auto r : spacers) {
        if (start_pos < r.first && signal_start / stride < seq_to_sig_map[r.first] / stride) {
            subreads.push_back(
                    subread(*read, {start_pos, r.first}, {signal_start, seq_to_sig_map[r.first]}));
        }
        start_pos = r.second;
        signal_start = seq_to_sig_map[r.second];
    }
    assert(read->raw_data.size(0) == seq_to_sig_map[read->seq.size()]);
    if (start_pos < read->seq.size() && signal_start / stride < read->raw_data.size(0) / stride) {
        subreads.push_back(subread(*read, {start_pos, read->seq.size()},
                                   {signal_start, read->raw_data.size(0)}));
    }

    return subreads;
}

std::vector<std::pair<std::string, DuplexSplitNode::SplitFinderF>>
DuplexSplitNode::build_split_finders() const {
    std::vector<std::pair<std::string, SplitFinderF>> split_finders;
    split_finders.push_back({"PORE_ADAPTER", [&](const ExtRead& read) {
                                 return filter_ranges(read.possible_pore_regions, [&](PosRange r) {
                                     return check_nearby_adapter(*read.read, r,
                                                                 m_settings.adapter_edist);
                                 });
                             }});

    if (!m_settings.simplex_mode) {
        split_finders.push_back(
                {"PORE_FLANK", [&](const ExtRead& read) {
                     return merge_ranges(
                             filter_ranges(read.possible_pore_regions,
                                           [&](PosRange r) {
                                               return check_flank_match(*read.read, r,
                                                                        m_settings.flank_err);
                                           }),
                             m_settings.strand_end_flank + m_settings.strand_start_flank);
                 }});

        split_finders.push_back(
                {"PORE_ALL", [&](const ExtRead& read) {
                     return merge_ranges(
                             filter_ranges(read.possible_pore_regions,
                                           [&](PosRange r) {
                                               return check_nearby_adapter(
                                                              *read.read, r,
                                                              m_settings.relaxed_adapter_edist) &&
                                                      check_flank_match(
                                                              *read.read, r,
                                                              m_settings.relaxed_flank_err);
                                           }),
                             m_settings.strand_end_flank + m_settings.strand_start_flank);
                 }});

        split_finders.push_back(
                {"ADAPTER_FLANK", [&](const ExtRead& read) {
                     return filter_ranges(find_adapter_matches(m_settings.adapter, read.read->seq,
                                                               m_settings.adapter_edist,
                                                               m_settings.expect_adapter_prefix),
                                          [&](PosRange r) {
                                              return check_flank_match(*read.read,
                                                                       {r.first, r.first},
                                                                       m_settings.flank_err);
                                          });
                 }});

        split_finders.push_back({"ADAPTER_MIDDLE", [&](const ExtRead& read) {
                                     if (auto split = identify_middle_adapter_split(*read.read)) {
                                         return PosRanges{*split};
                                     } else {
                                         return PosRanges();
                                     }
                                 }});

        split_finders.push_back({"SPLIT_MIDDLE", [&](const ExtRead& read) {
                                     if (auto split = identify_extra_middle_split(*read.read)) {
                                         return PosRanges{*split};
                                     } else {
                                         return PosRanges();
                                     }
                                 }});
    }

    return split_finders;
}

std::vector<std::shared_ptr<Read>> DuplexSplitNode::split(std::shared_ptr<Read> init_read) const {
    using namespace std::chrono;

    auto start_ts = high_resolution_clock::now();
    spdlog::trace("Processing read {}; length {}", init_read->read_id, init_read->seq.size());

    //assert(!init_read->seq.empty() && !init_read->moves.empty());
    if (init_read->seq.empty() || init_read->moves.empty()) {
        spdlog::trace("Empty read {}; length {}; moves {}", init_read->read_id,
                      init_read->seq.size(), init_read->moves.size());
        return std::vector<std::shared_ptr<Read>>{std::move(init_read)};
    }

    std::vector<ExtRead> to_split{create_ext_read(init_read)};
    for (const auto& [description, split_f] : m_split_finders) {
        spdlog::trace("Running {}", description);
        std::vector<ExtRead> split_round_result;
        for (auto& r : to_split) {
            auto spacers = split_f(r);
            spdlog::trace("DSN: {} strategy {} splits in read {}", description, spacers.size(),
                          init_read->read_id);

            if (spacers.empty()) {
                split_round_result.push_back(std::move(r));
            } else {
                for (auto sr : subreads(r.read, spacers)) {
                    split_round_result.push_back(create_ext_read(sr));
                }
            }
        }
        to_split = std::move(split_round_result);
    }

    std::vector<std::shared_ptr<Read>> split_result;
    size_t subread_id = 0;
    for (const auto& ext_read : to_split) {
        ext_read.read->subread_id = subread_id++;
        ext_read.read->split_count = to_split.size();
        split_result.push_back(std::move(ext_read.read));
    }

    spdlog::trace("Read {} split into {} subreads", init_read->read_id, split_result.size());

    auto stop_ts = high_resolution_clock::now();
    spdlog::trace("READ duration: {} microseconds (ID: {})",
                  duration_cast<microseconds>(stop_ts - start_ts).count(), init_read->read_id);

    return split_result;
}

void DuplexSplitNode::worker_thread() {
    m_active++;  // Track active threads.
    Message message;

    while (m_work_queue.try_pop(message)) {
        if (!m_settings.enabled) {
            send_message_to_sink(std::move(message));
        } else {
            // If this message isn't a read, we'll get a bad_variant_access exception.
            auto init_read = std::get<std::shared_ptr<Read>>(message);
            for (auto& subread : split(init_read)) {
                //TODO correctly process end_reason when we have them
                send_message_to_sink(std::move(subread));
            }
        }
    }

    --m_active;
}

DuplexSplitNode::DuplexSplitNode(DuplexSplitSettings settings,
                                 int num_worker_threads,
                                 size_t max_reads)
        : MessageSink(max_reads),
          m_settings(std::move(settings)),
          m_num_worker_threads(num_worker_threads) {
    m_split_finders = build_split_finders();
    for (int i = 0; i < m_num_worker_threads; i++) {
        worker_threads.push_back(
                std::make_unique<std::thread>(&DuplexSplitNode::worker_thread, this));
    }
}

void DuplexSplitNode::terminate_impl() {
    terminate_input_queue();

    // Wait for all the Node's worker threads to terminate
    for (auto& t : worker_threads) {
        if (t->joinable()) {
            t->join();
        }
    }
}

stats::NamedStats DuplexSplitNode::sample_stats() const { return stats::from_obj(m_work_queue); }

}  // namespace dorado
