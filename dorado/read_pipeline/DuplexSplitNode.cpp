#include "DuplexSplitNode.h"

#include "utils/alignment_utils.h"
#include "utils/duplex_utils.h"
#include "utils/sequence_utils.h"

#include <openssl/sha.h>
#include <spdlog/spdlog.h>

#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>

//To enable debug in build use "cmake -D CMAKE_CXX_FLAGS='-DDEBUG_READ_SPLITTING' -S . -B build"
#ifdef DEBUG_READ_SPLITTING
static constexpr bool DEBUG = true;
#else
static constexpr bool DEBUG = false;
#endif

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

std::shared_ptr<Read> copy_read(const Read& read) {
    auto copy = std::make_shared<Read>();
    copy->raw_data = read.raw_data;
    copy->digitisation = read.digitisation;
    copy->range = read.range;
    copy->offset = read.offset;
    copy->sample_rate = read.sample_rate;

    copy->shift = read.shift;
    copy->scale = read.scale;

    copy->scaling = read.scaling;

    copy->num_chunks = read.num_chunks;
    copy->num_modbase_chunks = read.num_modbase_chunks;

    copy->model_stride = read.model_stride;

    copy->read_id = read.read_id;
    copy->seq = read.seq;
    copy->qstring = read.qstring;
    copy->moves = read.moves;
    copy->run_id = read.run_id;
    copy->model_name = read.model_name;

    copy->base_mod_probs = read.base_mod_probs;
    copy->base_mod_info = read.base_mod_info;

    copy->num_trimmed_samples = read.num_trimmed_samples;

    copy->attributes = read.attributes;
    return copy;
}

//TODO copied from DataLoader.cpp
std::string get_string_timestamp_from_unix_time(time_t time_stamp_ms) {
    static std::mutex timestamp_mtx;
    std::unique_lock lock(timestamp_mtx);
    //Convert a time_t (seconds from UNIX epoch) to a timestamp in %Y-%m-%dT%H:%M:%S format
    auto time_stamp_s = time_stamp_ms / 1000;
    int num_ms = time_stamp_ms % 1000;
    char buf[32];
    struct tm ts;
    ts = *gmtime(&time_stamp_s);

    std::stringstream ss;
    ss << std::put_time(&ts, "%Y-%m-%dT%H:%M:%S.");
    ss << std::setfill('0') << std::setw(3) << num_ms;  // add ms
    ss << "+00:00";                                     //add zero timezone
    return ss.str();
}

// Expects the time to be encoded like "2017-09-12T9:50:12.456+00:00".
time_t get_unix_time_from_string_timestamp(const std::string& time_stamp) {
    std::stringstream ss(time_stamp);
    std::tm base_time = {};
    ss >> std::get_time(&base_time, "%Y-%m-%dT%H:%M:%S.");

    auto num_ms = std::stoi(time_stamp.substr(20, time_stamp.size() - 26));
    return mktime(&base_time) * 1000 + num_ms;
}

std::string adjust_time_ms(const std::string& time_stamp, uint64_t offset_ms) {
    return get_string_timestamp_from_unix_time(get_unix_time_from_string_timestamp(time_stamp) +
                                               offset_ms);
}

std::string derive_uuid(const std::string& input_uuid, const std::string& desc) {
    // Hash the input UUID using SHA-256
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, input_uuid.c_str(), input_uuid.size());
    SHA256_Update(&sha256, desc.c_str(), desc.size());
    SHA256_Final(hash, &sha256);

    // Truncate the hash to 16 bytes (128 bits) to match the size of a UUID
    std::array<unsigned char, 16> truncated_hash;
    std::copy(std::begin(hash), std::begin(hash) + 16, std::begin(truncated_hash));

    // Set the UUID version to 4 (random)
    truncated_hash[6] = (truncated_hash[6] & 0x0F) | 0x40;

    // Set the UUID variant to the RFC 4122 specified value (10)
    truncated_hash[8] = (truncated_hash[8] & 0x3F) | 0x80;

    // Convert the truncated hash to a UUID string
    std::stringstream ss;
    for (size_t i = 0; i < truncated_hash.size(); ++i) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(truncated_hash[i]);
        if (i == 3 || i == 5 || i == 7 || i == 9) {
            ss << "-";
        }
    }

    return ss.str();
}

//merges overlapping ranges and ranges separated by merge_dist or less
//ranges supposed to be sorted by start coordinate
PosRanges merge_ranges(const PosRanges& ranges, size_t merge_dist) {
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

std::vector<std::pair<size_t, size_t>> detect_pore_signal(const torch::Tensor& signal,
                                                          float threshold,
                                                          size_t cluster_dist,
                                                          size_t ignore_prefix) {
    std::vector<std::pair<size_t, size_t>> ans;
    auto pore_a = signal.accessor<float, 1>();
    size_t cl_start = size_t(-1);
    size_t cl_end = size_t(-1);

    for (size_t i = ignore_prefix; i < pore_a.size(0); i++) {
        if (pore_a[i] > threshold) {
            //check if we need to start new cluster
            if (cl_end == size_t(-1) || i > cl_end + cluster_dist) {
                //report previous cluster
                if (cl_end != size_t(-1)) {
                    assert(cl_start != size_t(-1));
                    ans.push_back({cl_start, cl_end});
                }
                cl_start = i;
            }
            cl_end = i + 1;
        }
    }
    //report last cluster
    if (cl_end != size_t(-1)) {
        assert(cl_start < pore_a.size(0) && cl_end <= pore_a.size(0));
        assert(cl_start != size_t(-1));
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

    auto edlib_cfg = [&] {
        if constexpr (DEBUG) {
            return edlibNewAlignConfig(-1, EDLIB_MODE_HW, EDLIB_TASK_PATH, NULL, 0);
        } else {
            return edlibNewAlignConfig(dist_thr, EDLIB_MODE_HW, EDLIB_TASK_LOC, NULL, 0);
        }
    }();

    auto edlib_result =
            edlibAlign(adapter.c_str(), adapter.size(), seq.c_str() + shift, span, edlib_cfg);
    assert(edlib_result.status == EDLIB_STATUS_OK);
    std::optional<PosRange> res = std::nullopt;
    if (edlib_result.status == EDLIB_STATUS_OK && edlib_result.editDistance != -1) {
        if constexpr (DEBUG) {
            spdlog::debug("Finding adapter in range: [{}, {}]", subrange.first, subrange.second),
                    spdlog::debug("Best adapter match edit distance: {} ; is middle {}",
                                  edlib_result.editDistance,
                                  abs(int(span / 2) - edlib_result.startLocations[0]) < 1000);
            spdlog::debug("Match location: ({}, {})", edlib_result.startLocations[0] + shift,
                          edlib_result.endLocations[0] + shift + 1);
            spdlog::debug("\n{}", utils::print_alignment(adapter.c_str(), seq.c_str() + shift,
                                                         edlib_result));
            if (edlib_result.editDistance <= dist_thr) {
                res = {edlib_result.startLocations[0] + shift,
                       edlib_result.endLocations[0] + shift + 1};
            }
        } else {
            assert(edlib_result.editDistance <= dist_thr);
            res = {edlib_result.startLocations[0] + shift,
                   edlib_result.endLocations[0] + shift + 1};
        }
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
bool check_rc_match(const std::string& seq, PosRange templ_r, PosRange compl_r, int dist_thr) {
    assert(templ_r.second > templ_r.first);
    assert(compl_r.second > compl_r.first);
    assert(dist_thr >= 0);

    auto rc_compl = dorado::utils::reverse_complement(
            seq.substr(compl_r.first, compl_r.second - compl_r.first));

    auto edlib_cfg = [&] {
        if constexpr (DEBUG) {
            return edlibNewAlignConfig(-1, EDLIB_MODE_HW, EDLIB_TASK_PATH, NULL, 0);
        } else {
            return edlibNewAlignConfig(dist_thr, EDLIB_MODE_HW, EDLIB_TASK_DISTANCE, NULL, 0);
        }
    }();

    const char* c_seq = seq.c_str();
    auto edlib_result = edlibAlign(c_seq + templ_r.first, templ_r.second - templ_r.first,
                                   rc_compl.c_str(), rc_compl.size(), edlib_cfg);
    assert(edlib_result.status == EDLIB_STATUS_OK);
    std::optional<PosRange> res = std::nullopt;

    bool match = (edlib_result.status == EDLIB_STATUS_OK) && (edlib_result.editDistance != -1);
    if constexpr (DEBUG) {
        spdlog::debug("Checking ranges [{}, {}] vs [{}, {}]: edist={}\n{}", templ_r.first,
                      templ_r.second, compl_r.first, compl_r.second, edlib_result.editDistance,
                      utils::print_alignment(c_seq + templ_r.first, rc_compl.data(), edlib_result));
        match = match && (edlib_result.editDistance <= dist_thr);
    } else {
        assert(!match || edlib_result.editDistance <= dist_thr);
    }

    edlibFreeAlignResult(edlib_result);
    return match;
}

//TODO end_reason access?
//signal_range should already be 'adjusted' to stride (e.g. probably gotten from seq_range)
//NB: doesn't set parent_read_id
std::shared_ptr<Read> subread(const Read& read, PosRange seq_range, PosRange signal_range) {
    //TODO support mods
    //NB: currently doesn't support mods
    assert(read.base_mod_info == nullptr && read.base_mod_probs.empty());
    const int stride = read.model_stride;
    assert(signal_range.first % stride == 0);
    assert(signal_range.second % stride == 0 ||
           (signal_range.second == read.raw_data.size(0) && seq_range.second == read.seq.size()));

    //assert(read.called_chunks.empty() && read.num_chunks_called == 0 && read.num_modbase_chunks_called == 0);
    auto subread = copy_read(read);

    //TODO is it ok, or do we want subread number here?
    const auto subread_id = derive_uuid(
            read.read_id, std::to_string(seq_range.first) + "-" + std::to_string(seq_range.second));
    subread->read_id = subread_id;
    subread->raw_data = subread->raw_data.index(
            {torch::indexing::Slice(signal_range.first, signal_range.second)});
    subread->attributes.read_number = uint32_t(-1);
    subread->attributes.start_time =
            adjust_time_ms(subread->attributes.start_time,
                           uint64_t(std::round((subread->num_trimmed_samples + signal_range.first) *
                                               1000. / subread->sample_rate)));
    //we adjust for it in new start time above
    subread->num_trimmed_samples = 0;
    ////FIXME update?
    //subread->range = ???;
    ////FIXME update?
    //subread->offset = ???;

    subread->seq = subread->seq.substr(seq_range.first, seq_range.second - seq_range.first);
    subread->qstring = subread->qstring.substr(seq_range.first, seq_range.second - seq_range.first);
    subread->moves = std::vector<uint8_t>(subread->moves.begin() + signal_range.first / stride,
                                          subread->moves.begin() + signal_range.second / stride);
    assert(signal_range.second == read.raw_data.size(0) ||
           subread->moves.size() * stride == subread->raw_data.size(0));
    return subread;
}

}  // namespace

namespace dorado {

std::vector<uint64_t> move_cum_sums(const std::vector<uint8_t> moves) {
    std::vector<uint64_t> ans(moves.size(), 0);
    if (!moves.empty()) {
        ans[0] = moves[0];
    }
    for (size_t i = 1, n = moves.size(); i < n; i++) {
        ans[i] = ans[i - 1] + moves[i];
    }
    return ans;
}

DuplexSplitNode::ExtRead::ExtRead(std::shared_ptr<Read> r)
        : read(std::move(r)),
          data_as_float32(read->raw_data.to(torch::kFloat)),
          move_sums(move_cum_sums(read->moves)) {
    //TODO switch to inclusive_scan and remove manual cumulative sums implementation
    // when arm64:bionic compiler really supports c++17
    //      move_sums(read->moves.size(), 0) {
    //Using std::inclusive_scan instead of std::partial_sum since the latter overflows
    //std::inclusive_scan(read->moves.begin(), read->moves.end(), move_sums.begin(), std::plus(),
    //                    uint64_t(0));
    assert(move_sums.back() == read->seq.length());
}

PosRanges DuplexSplitNode::possible_pore_regions(const DuplexSplitNode::ExtRead& read,
                                                 float pore_thr) const {
    PosRanges pore_regions;

    //pA formula before scaling:
    //pA = read->scaling * (raw + read->offset);
    //pA formula after scaling:
    //pA = read->scale * raw + read->shift
    spdlog::debug("Analyzing signal in read {}", read.read->read_id);

    if constexpr (DEBUG) {
        const auto max_val =
                (m_settings.expect_pore_prefix >= read.read->raw_data.size(0))
                        ? 0.0f
                        : (read.data_as_float32 * read.read->scale + read.read->shift)
                                  .index({torch::indexing::Slice(m_settings.expect_pore_prefix,
                                                                 torch::indexing::None)})
                                  .max()
                                  .item<float>();
        spdlog::debug("Max raw signal {} pA, threshold: {}", max_val, pore_thr);
    }

    auto pore_sample_ranges = detect_pore_signal(
            read.data_as_float32, (pore_thr - read.read->shift) / read.read->scale,
            m_settings.pore_cl_dist, m_settings.expect_pore_prefix);

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
        pore_regions.push_back({start_pos, end_pos});
    }

    if constexpr (DEBUG) {
        std::ostringstream oss;
        std::copy(pore_regions.begin(), pore_regions.end(),
                  std::ostream_iterator<PosRange>(oss, "; "));
        spdlog::debug("{} regions to check: {}", pore_regions.size(), oss.str());
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

//r is potential spacer region
bool DuplexSplitNode::check_flank_match(const Read& read, PosRange r, int dist_thr) const {
    return r.first >= m_settings.end_flank &&
           r.second + m_settings.start_flank <= read.seq.length() &&
           check_rc_match(read.seq, {r.first - m_settings.end_flank, r.first - m_settings.end_trim},
                          //including spacer region in search
                          {r.first, r.second + m_settings.start_flank}, dist_thr);
}

std::optional<DuplexSplitNode::PosRange> DuplexSplitNode::identify_extra_middle_split(
        const Read& read) const {
    const auto r_l = read.seq.size();
    const auto search_span = std::max(m_settings.middle_adapter_search_span,
                                      int(std::round(m_settings.middle_adapter_search_frac * r_l)));
    if (r_l < m_settings.end_flank + m_settings.start_flank || r_l < search_span) {
        return std::nullopt;
    }

    spdlog::trace("Searching for adapter match");
    if (auto adapter_match = find_best_adapter_match(
                m_settings.adapter, read.seq, m_settings.relaxed_adapter_edist,
                {r_l / 2 - search_span / 2, r_l / 2 + search_span / 2})) {
        auto adapter_start = adapter_match->first;
        spdlog::trace("Checking middle match & start/end match");
        if (check_flank_match(read, {adapter_start, adapter_start},
                              m_settings.relaxed_flank_edist) &&
            check_rc_match(read.seq, {r_l - m_settings.end_flank, r_l - m_settings.end_trim},
                           {0, m_settings.start_flank}, m_settings.relaxed_flank_edist)) {
            return PosRange{adapter_start - 1, adapter_start};
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

    const auto seq_to_sig_map = utils::moves_to_map(read->moves, read->model_stride,
                                                    read->raw_data.size(0), read->seq.size() + 1);

    //TODO maybe simplify by adding begin/end stubs?
    uint64_t start_pos = 0;
    uint64_t signal_start = seq_to_sig_map[0];
    for (auto r : spacers) {
        subreads.push_back(
                subread(*read, {start_pos, r.first}, {signal_start, seq_to_sig_map[r.first]}));
        start_pos = r.second;
        signal_start = seq_to_sig_map[r.second];
    }
    assert(read->raw_data.size(0) == seq_to_sig_map[read->seq.size()]);
    subreads.push_back(
            subread(*read, {start_pos, read->seq.size()}, {signal_start, read->raw_data.size(0)}));

    if constexpr (DEBUG) {
        std::ostringstream spacer_oss;
        std::copy(spacers.begin(), spacers.end(),
                  std::ostream_iterator<PosRange>(spacer_oss, "; "));

        std::ostringstream id_oss;
        std::transform(subreads.begin(), subreads.end(),
                       std::ostream_iterator<std::string>(id_oss, "; "),
                       [](std::shared_ptr<Read> sr) { return sr->read_id; });
        spdlog::debug("{} spacing regions in subread {}: {}. New read ids: {}", spacers.size(),
                      read->read_id, spacer_oss.str(), id_oss.str());
    }

    return subreads;
}

std::vector<std::pair<std::string, DuplexSplitNode::SplitFinderF>>
DuplexSplitNode::build_split_finders() const {
    std::vector<std::pair<std::string, SplitFinderF>> split_finders;
    split_finders.push_back(
            {"PORE_ADAPTER", [&](const ExtRead& read) {
                 return filter_ranges(
                         possible_pore_regions(read, m_settings.pore_thr), [&](PosRange r) {
                             return check_nearby_adapter(*read.read, r, m_settings.adapter_edist);
                         });
             }});

    if (!m_settings.simplex_mode) {
        split_finders.push_back(
                {"PORE_FLANK", [&](const ExtRead& read) {
                     return merge_ranges(
                             filter_ranges(possible_pore_regions(read, m_settings.pore_thr),
                                           [&](PosRange r) {
                                               return check_flank_match(*read.read, r,
                                                                        m_settings.flank_edist);
                                           }),
                             m_settings.end_flank + m_settings.start_flank);
                 }});

        split_finders.push_back(
                {"PORE_ALL", [&](const ExtRead& read) {
                     return merge_ranges(
                             filter_ranges(possible_pore_regions(read, m_settings.relaxed_pore_thr),
                                           [&](PosRange r) {
                                               return check_nearby_adapter(
                                                              *read.read, r,
                                                              m_settings.relaxed_adapter_edist) &&
                                                      check_flank_match(
                                                              *read.read, r,
                                                              m_settings.relaxed_flank_edist);
                                           }),
                             m_settings.end_flank + m_settings.start_flank);
                 }});

        split_finders.push_back(
                {"ADAPTER_FLANK", [&](const ExtRead& read) {
                     return filter_ranges(find_adapter_matches(m_settings.adapter, read.read->seq,
                                                               m_settings.adapter_edist,
                                                               m_settings.expect_adapter_prefix),
                                          [&](PosRange r) {
                                              return check_flank_match(*read.read,
                                                                       {r.first, r.first},
                                                                       m_settings.flank_edist);
                                          });
                 }});

        split_finders.push_back({"ADAPTER_MIDDLE", [&](const ExtRead& read) {
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
    spdlog::debug("Processing read {}; length {}", init_read->read_id, init_read->seq.size());

    std::vector<ExtRead> to_split{ExtRead(init_read)};
    for (const auto& [description, split_f] : m_split_finders) {
        spdlog::trace("Running {}", description);
        std::vector<ExtRead> split_round_result;
        for (auto& r : to_split) {
            //auto start = high_resolution_clock::now();
            auto spacers = split_f(r);
            //auto stop = high_resolution_clock::now();
            //spdlog::trace("{} duration: {} microseconds", description, duration_cast<microseconds>(stop - start).count());
            spdlog::debug("DSN: {} strategy {} splits in read {}", description, spacers.size(),
                          init_read->read_id);

            if (spacers.empty()) {
                split_round_result.push_back(std::move(r));
            } else {
                for (auto sr : subreads(r.read, spacers)) {
                    split_round_result.emplace_back(sr);
                }
            }
        }
        to_split = std::move(split_round_result);
    }

    std::vector<std::shared_ptr<Read>> split_result;
    for (const auto& er : to_split) {
        split_result.push_back(std::move(er.read));
    }

    spdlog::debug("Read {} split into {} subreads", init_read->read_id, split_result.size());

    auto stop_ts = high_resolution_clock::now();
    spdlog::trace("READ duration: {} microseconds (ID: {})",
                  duration_cast<microseconds>(stop_ts - start_ts).count(), init_read->read_id);

    return split_result;
}

void DuplexSplitNode::worker_thread() {
    Message message;

    while (m_work_queue.try_pop(message)) {
        if (!m_settings.enabled) {
            m_sink.push_message(std::move(message));
        } else {
            // If this message isn't a read, we'll get a bad_variant_access exception.
            auto init_read = std::get<std::shared_ptr<Read>>(message);
            for (auto& subread : split(init_read)) {
                //TODO correctly process end_reason when we have them
                if (subread->read_id != init_read->read_id) {
                    subread->parent_read_id = init_read->read_id;
                }
                m_sink.push_message(std::move(subread));
            }
        }
    }
    m_sink.terminate();
}

DuplexSplitNode::DuplexSplitNode(MessageSink& sink,
                                 DuplexSplitSettings settings,
                                 int num_worker_threads,
                                 size_t max_reads)
        : MessageSink(max_reads),
          m_sink(sink),
          m_settings(std::move(settings)),
          m_num_worker_threads(num_worker_threads) {
    m_split_finders = build_split_finders();
    for (int i = 0; i < m_num_worker_threads; i++) {
        worker_threads.push_back(
                std::make_unique<std::thread>(&DuplexSplitNode::worker_thread, this));
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

}  // namespace dorado
