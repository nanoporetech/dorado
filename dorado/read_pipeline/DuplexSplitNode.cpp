#include "DuplexSplitNode.h"

#include "utils/sequence_utils.h"
#include "utils/duplex_utils.h"
#include "3rdparty/edlib/edlib/include/edlib.h"

#include <spdlog/spdlog.h>
#include <optional>
#include <cmath>

#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <array>
#include <openssl/sha.h>

namespace {

using namespace dorado;

typedef DuplexSplitNode::PosRange PosRange;

std::ostream& operator<<(std::ostream& os, const PosRange& r)
{
    return os << "[" << r.first << ", " << r.second << "]";
}

//FIXME ask if we can have some copy constructor?
std::shared_ptr<Read> copy_read(const Read &read) {
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
    copy->base_mod_probs = read.base_mod_probs;
    copy->run_id = read.run_id;
    copy->model_name = read.model_name;

    copy->base_mod_info = read.base_mod_info;

    copy->num_trimmed_samples = read.num_trimmed_samples;

    copy->attributes = read.attributes;
    return copy;
}

//FIXME copied from DataLoader.cpp
std::string get_string_timestamp_from_unix_time(time_t time_stamp_ms) {
    static std::mutex timestamp_mtx;
    std::unique_lock lock(timestamp_mtx);
    //Convert a time_t (seconds from UNIX epoch) to a timestamp in %Y-%m-%dT%H:%M:%S format
    auto time_stamp_s = time_stamp_ms / 1000;
    int num_ms = time_stamp_ms % 1000;
    char buf[32];
    struct tm ts;
    ts = *gmtime(&time_stamp_s);
    strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S.", &ts);
    std::string time_stamp_str = std::string(buf);
    time_stamp_str += std::to_string(num_ms);  // add ms
    time_stamp_str += "+00:00";                //add zero timezone
    return time_stamp_str;
}

// Expects the time to be encoded like "2017-09-12T9:50:12.456+00:00".
time_t get_unix_time_from_string_timestamp(const std::string& time_stamp) {
    static std::mutex timestamp_mtx;
    std::unique_lock lock(timestamp_mtx);
    std::tm base_time = {};
    strptime(time_stamp.c_str(), "%Y-%m-%dT%H:%M:%S.", &base_time);
    auto num_ms = std::stoi(time_stamp.substr(20, time_stamp.size()-26));
    return mktime(&base_time) * 1000 + num_ms;
}

std::string adjust_time_ms(const std::string& time_stamp, uint64_t offset_ms) {
    return get_string_timestamp_from_unix_time(
                get_unix_time_from_string_timestamp(time_stamp)
                + offset_ms);
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

std::vector<std::pair<size_t, size_t>> detect_pore_signal(torch::Tensor pa_signal,
                                                          float threshold,
                                                          size_t cluster_dist) {
    spdlog::info("Max raw signal {} pA", pa_signal.max().item<float>());

    std::vector<std::pair<size_t, size_t>> ans;
    //FIXME what type to use here?
    auto pore_a = pa_signal.accessor<float, 1>();
    size_t start = 0;
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
    assert(start < pore_a.size(0) && end <= pore_a.size(0));
    return ans;
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
            spdlog::info("Best adapter match edit distance: {} ; is middle {}",
                        edlib_result.editDistance, abs(int(span / 2) - edlib_result.startLocations[0]) < 1000);
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

std::string print_alignment(const char* query, const char* target, const EdlibAlignResult& result) {
    std::stringstream ss;
    int tpos = result.startLocations[0];

    int qpos = 0;
    for (int i = 0; i < result.alignmentLength; i++) {
        if (result.alignment[i] == EDLIB_EDOP_DELETE) {
            ss << "-";
        } else {
            ss << query[qpos];
            qpos++;
        }
    }

    ss << '\n';

    for (int i = 0; i < result.alignmentLength; i++) {
        if (result.alignment[i] == EDLIB_EDOP_MATCH) {
            ss << "|";
        } else if (result.alignment[i] == EDLIB_EDOP_INSERT) {
            ss << " ";
        } else if (result.alignment[i] == EDLIB_EDOP_DELETE) {
            ss << " ";
        } else if (result.alignment[i] == EDLIB_EDOP_MISMATCH) {
            ss << "*";
        }
    }

    ss << '\n';

    for (int i = 0; i < result.alignmentLength; i++) {
        if (result.alignment[i] == EDLIB_EDOP_INSERT) {
            ss << "-";
        } else {
            ss << target[tpos];
            tpos++;
        }
    }

    return ss.str();
}

bool check_rc_match(const std::string& seq, PosRange templ_r, PosRange compl_r, int dist_thr) {
    assert(templ_r.second > templ_r.first && compl_r.second > compl_r.first);
    const char* c_seq = seq.c_str();
    std::vector<char> rc_compl(c_seq + compl_r.first, c_seq + compl_r.second);
    dorado::utils::reverse_complement(rc_compl);

    auto edlib_result = edlibAlign(c_seq + templ_r.first,
                            templ_r.second - templ_r.first,
                            rc_compl.data(), rc_compl.size(),
                            //edlibNewAlignConfig(-1, EDLIB_MODE_HW, EDLIB_TASK_DISTANCE, NULL, 0));
                            edlibNewAlignConfig(-1, EDLIB_MODE_HW, EDLIB_TASK_PATH, NULL, 0));
    assert(edlib_result.status == EDLIB_STATUS_OK);
    std::optional<PosRange> res = std::nullopt;
    assert(edlib_result.status == EDLIB_STATUS_OK && edlib_result.editDistance <= dist_thr);
    spdlog::info("Checking ranges [{}, {}] vs [{}, {}]: edist={}\n{}",
                    templ_r.first, templ_r.second,
                    compl_r.first, compl_r.second,
                    edlib_result.editDistance,
                    print_alignment(c_seq + templ_r.first, rc_compl.data(), edlib_result));

    //FIXME integrate dist_thr check right into align call after tweaking the settings
    bool match = (edlib_result.status == EDLIB_STATUS_OK)
                    && (edlib_result.editDistance != -1)
                    && (edlib_result.editDistance <= dist_thr);
    edlibFreeAlignResult(edlib_result);
    return match;
}

//TODO end_reason access?
//signal_range should already be 'adjusted' to stride (e.g. probably gotten from seq_range)
std::shared_ptr<Read> subread(const Read& read, PosRange seq_range, PosRange signal_range) {
    const int stride = read.model_stride;
    assert(signal_range.first % stride == 0);
    assert(signal_range.second % stride == 0 || (signal_range.second == read.raw_data.size(0) && seq_range.second == read.seq.size()));

    //assert(read.called_chunks.empty() && read.num_chunks_called == 0 && read.num_modbase_chunks_called == 0);
    auto subread = copy_read(read);

    //TODO is it ok, or do we want subread number here?
    const auto subread_id = derive_uuid(read.read_id,
                        std::to_string(seq_range.first) + "-" + std::to_string(seq_range.second));
    subread->read_id = subread_id;
    subread->raw_data = subread->raw_data.index({torch::indexing::Slice(signal_range.first, signal_range.second)});
    subread->attributes.start_time = adjust_time_ms(subread->attributes.start_time,
                                        (subread->num_trimmed_samples + signal_range.first)
                                            * 1000. / subread->sample_rate);
    //we adjust for it in new start time above
    subread->num_trimmed_samples = 0;
    //FIXME HOW TO UPDATE
    //subread->attributes.read_number = ???;
    ////fixme update?
    //subread->range = ???;
    ////fixme update?
    //subread->offset = ???;

    subread->seq = subread->seq.substr(seq_range.first, seq_range.second - seq_range.first);
    subread->qstring = subread->qstring.substr(seq_range.first, seq_range.second - seq_range.first);
    subread->moves = std::vector<uint8_t>(subread->moves.begin() + signal_range.first / stride,
        subread->moves.begin() + signal_range.second / stride);
    assert(signal_range.second == read.raw_data.size(0) || subread->moves.size() * stride == subread->raw_data.size(0));
    return subread;
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
    spdlog::info("Analyzing signal in read {}", read.read_id);
    for (auto pore_signal_region : detect_pore_signal(
                                   read.raw_data.to(torch::kFloat) * read.scale + read.shift,
                                   m_settings.pore_thr,
                                   m_settings.pore_cl_dist)) {
        auto move_start = pore_signal_region.first / read.model_stride;
        auto move_end = pore_signal_region.second / read.model_stride;
        assert(move_end >= move_start);
        //NB move_start can get to move_sums.size(), because of the stride rounding?
        if (move_start >= move_sums.size() || move_end >= move_sums.size() || move_sums.at(move_start) == 0) {
            //either at very end of the signal or basecalls have not started yet
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
    spdlog::info("DSN: Finding adapter matches in read {}", read.read_id);
    auto adapter_region_candidates = find_adapter_matches(m_settings.adapter, read.seq,
                                            m_settings.adapter_edist,
                                            m_settings.expect_adapter_prefix);

    auto [definite_splits, extra_regions] =
                match_pore_adapter(pore_region_candidates,
                                   adapter_region_candidates,
                                   m_settings.pore_adapter_gap_thr);
    auto matched_pore_adapter = definite_splits.size();

    //checking reverse-complement matching for uncertain regions
    //No need to 'cut' adapter region -- matching complement region
    // won't be penalized for having extra prefix & suffix in edlib
    for (auto r : extra_regions) {
        if (r.first < m_settings.templ_flank ||
            r.second + m_settings.compl_flank > read.seq.length()) {
            //TODO maybe handle some of these cases too (extra min_avail_sequence parameter?)
            continue;
        }
        //FIXME any need to adjust for adapter
        //FIXME should we subtract a portion of tail adapter from the first region too?
        if (check_rc_match(read.seq,
                        {r.first - m_settings.templ_flank, r.first - m_settings.templ_trim},
                        {r.second, r.second + m_settings.compl_flank},
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
    assert(read.raw_data.size(0) == seq_to_sig_map[read.seq.size()]);
    ans.push_back(subread(read, {start_pos, read.seq.size()}, {signal_start, read.raw_data.size(0)}));
    return ans;
}

void DuplexSplitNode::worker_thread() {
    spdlog::info("DSN: Hello from worker thread");
    Message message;
    while (m_work_queue.try_pop(message)) {
        // If this message isn't a read, we'll get a bad_variant_access exception.
        auto read = std::get<std::shared_ptr<Read>>(message);

        auto interspace_ranges = identify_splits(*read);
        std::ostringstream oss;
        std::copy(interspace_ranges.begin(), interspace_ranges.end(), std::ostream_iterator<PosRange>(oss, ";"));
        for (auto r : interspace_ranges) {
            spdlog::info("BED\t{}\t{}\t{}", read->read_id, r.first, r.second);
        }

        spdlog::info("DSN: identified {} splits in read {}: {}", interspace_ranges.size(), read->read_id, oss.str());
        if (interspace_ranges.empty()) {
            m_sink.push_message(read);
        } else {
            for (auto m : split(*read, interspace_ranges)) {
                m_sink.push_message(std::move(m));
            }
        }
        //FIXME Just passes the reads to get basecalls for now
        //m_sink.push_message(std::move(message));
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