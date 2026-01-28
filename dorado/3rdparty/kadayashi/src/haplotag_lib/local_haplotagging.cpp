#include "local_haplotagging.h"

#include "bam_record_parsing.h"
#include "blocked_bloom_filter.h"
#include "faidx_utils.h"
#include "hts_types.h"
#include "sequence_utility.h"
#include "variant_graph.h"

#include <htslib/bgzf.h>
#include <htslib/faidx.h>
#include <htslib/hts.h>
#include <htslib/kfunc.h>
#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iterator>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>

#ifdef NDEBUG
#define LOG_TRACE(...)
#define LOG_DEBUG(...)
#else
#define LOG_TRACE(...) spdlog::trace(__VA_ARGS__)
#define LOG_DEBUG(...) spdlog::debug(__VA_ARGS__)
#endif

namespace kadayashi {

constexpr uint32_t MAX_READS = 134217727;  // note: need to modify pg_t to increase this

constexpr int TRF_MIN_TANDAM_DEPTH = 1;
constexpr int TRF_MOTIF_MAX_LEN = 200;
constexpr int TRF_ADD_PADDING = 10;
constexpr int TRF_CLOSE_GAP_THRESHOLD = 50;

constexpr char SENTINEL_REF_ALLELE[] = "M";
constexpr int SENTINEL_REF_ALLELE_L = 1;

namespace {

void add_allele_qa_v_nt4seq(std::vector<qa_t> &h,
                            const uint32_t pos,
                            const std::vector<uint8_t> &allele,
                            const int allele_l,
                            const uint8_t cigar_op) {
    h.push_back(qa_t{.pos = pos,
                     .allele = {},
                     .var_idx = std::numeric_limits<uint32_t>::max(),
                     .allele_idx = std::numeric_limits<uint32_t>::max(),
                     .is_used = 0,
                     .hp = HAPTAG_UNPHASED});
    h.back().allele.insert(h.back().allele.end(), allele.begin(), allele.begin() + allele_l);

    // append cigar operation to the allele integer sequence
    h.back().allele.push_back(cigar_op);
}

void filter_lift_qa_v_given_conf_list(const std::vector<qa_t> &src,
                                      std::vector<qa_t> &dst,
                                      const variants_t &ht_refvars) {
    // assumes variants in src are sorted by position.

    if (ht_refvars.empty() || src.empty()) {
        return;
    }

    for (size_t ir = 0; ir < src.size(); ir++) {
        const uint32_t pos = src[ir].pos;
        if (ht_refvars.find(pos) != ht_refvars.end()) {
            const qa_t &h = src[ir];

            // push
            const std::string_view sentinel_ref_allele{SENTINEL_REF_ALLELE};
            kadayashi::add_allele_qa_v(dst, h.pos, sentinel_ref_allele, VAR_OP_X);

            // copy over the actual allele sequence
            auto &a = dst.back().allele;
            a.clear();
            a.reserve(h.allele.size());
            for (size_t j = 0; j < h.allele.size(); j++) {
                a.emplace_back(h.allele[j]);
            }
        }
    }
}

std::vector<uint64_t> TRF_heuristic(std::string_view seq, const int ref_start) {
    // A simple tandem repeat masker inspired by TRF.
    // Uses arbitrary thresholds and does not exclude
    // non-tandem direct repeats, unlike the TRF.
    // The usecase is to merely avoid picking up variants
    // too close to low-complexity runs for phasing.
    const int seq_l = static_cast<int>(seq.size());

    // We will enumerate all k-mer combos and check for exact match.
    constexpr int K = 3;

    // input sancheck
    if (seq_l < K * 3) {  // sequence too short, nothing to do
        return {};
    }

    // init buffer for kmer index
    uint32_t idx_l = 1;
    for (int i = 0; i < K; i++) {
        idx_l *= 4;
    }
    std::vector<std::vector<uint32_t>> idx(idx_l);
    std::vector<uint8_t> idx_good(idx_l, 0);

    // index kmers
    const uint32_t mermask = (1 << (K * 2)) - 1;
    uint32_t mer = 0;
    int mer_n = 0;
    for (int i = 0; i < seq_l; i++) {
        const int c = (int)kdy_seq_nt4_table[(uint8_t)seq[i]];
        if (c != 4) {
            mer = (mer << 2 | c) & mermask;
            if (mer_n < K) {
                mer_n++;
            }
            if (mer_n == K) {
                idx[mer].push_back(i + ref_start - K + 1);  // use absolute coordinate
            }
        } else {
            mer = 0;
            mer_n = 0;
        }
    }

    // calculate distances between matches
    std::vector<uint32_t> ds0;  // collects non-singleton kmer match intervals from different kmers
    std::vector<std::vector<uint32_t>> dists(idx_l);  // collects kmer match intervals of each kmer
    for (uint32_t i = 0; i < idx_l; i++) {
        if (idx[i].size() >= 3) {
            const int l = static_cast<int>(idx[i].size());
            idx_good[i] = 1;
            dists[i].resize(l, 0);
            for (int j = 1; j < l; j++) {
                dists[i][j] = idx[i][j] - idx[i][j - 1];
            }

            // sort interval sizes
            std::sort(dists[i].begin(), dists[i].end());

            // Is there any recurring interval sizes?
            // if yes, remember them; otherwise blacklist the kmer.
            int ok = 0;
            int new_ = 1;
            for (int j = 2; j < l; j++) {
                if (dists[i][j] == dists[i][j - 1] && dists[i][j] == dists[i][j - 2]) {
                    if (new_) {
                        ok = 1;
                        new_ = 0;
                        ds0.push_back(dists[i][j]);
                    }
                }
            }
            if (!ok) {
                idx_good[i] = 0;
            } else {  // looks ok, restore the order in the buffer, we will use it again
                for (int j = 1; j < l; j++) {
                    dists[i][j] = idx[i][j] - idx[i][j - 1];
                }
            }
        }
    }

    // find promising match interval sizes
    std::vector<uint32_t> ds;
    std::sort(ds0.begin(), ds0.end());
    int cnt = 0;
    for (size_t i = 1; i < ds0.size(); i++) {
        if (ds0[i] != ds0[i - 1]) {
            if (cnt > 0) {
                if (ds0[i - 1] < TRF_MOTIF_MAX_LEN) {
                    ds.push_back(ds0[i - 1]);
                }
            }
            cnt = 0;
        } else {
            cnt++;
        }
    }
    if ((cnt > 0) && (!ds0.empty()) && (ds0.back() < TRF_MOTIF_MAX_LEN)) {
        ds.push_back(ds0.back());
    }
    std::vector<uint64_t> intervals;
    if (ds.empty()) {
        if constexpr (DEBUG_LOCAL_HAPLOTAGGING) {
            LOG_TRACE("[kdys::{}] ds is empty (ds0 was {})", __func__, ds0.size());
            for (size_t i = 0; i < ds0.size(); i++) {
                LOG_TRACE("[kdys::{}] ds0 #{} is {}", __func__, i, ds0[i]);
            }
        }

    } else {
        if constexpr (DEBUG_LOCAL_HAPLOTAGGING) {
            LOG_TRACE("[kdys::{}] there are {} candidate distances (raw: {})", __func__, ds.size(),
                      ds0.size());
            for (size_t i = 0; i < ds.size(); i++) {
                LOG_TRACE("[kdys::{}] ds #{} is {}", __func__, i, ds[i]);
            }
        }

        // process the kmer chains
        std::vector<uint32_t> poss;
        std::vector<uint32_t> intervals_buf;
        std::vector<uint8_t> used(idx_l);
        int buf[3] = {0, 0, 0};
        bool failed = false;
        for (const uint32_t d : ds) {
            for (uint32_t i_mer = 0; i_mer < idx_l; i_mer++) {
                if (!idx_good[i_mer]) {
                    continue;
                }

                uint8_t mer_is_used = 0;

                poss.clear();
                int ok = 0;
                int stop = 0;
                for (int64_t i = 0; i < std::ssize(idx[i_mer]);
                     i++) {  // require at least 1 hit for three consecutive kmers
                    if (i + 3 < std::ssize(idx[i_mer])) {
                        buf[0] = dists[i_mer][i + 1] == d;
                        buf[1] = dists[i_mer][i + 2] == d;
                        buf[2] = dists[i_mer][i + 3] == d;
                        ok = buf[0] || buf[1] || buf[2];
                        buf[0] = dists[i_mer][i + 1] >= d && dists[i_mer][i + 1] < 3 * d + 3;
                        buf[1] = dists[i_mer][i + 2] >= d && dists[i_mer][i + 2] < 3 * d + 3;
                        buf[2] = dists[i_mer][i + 3] >= d && dists[i_mer][i + 3] < 3 * d + 3;
                        ok = ok & buf[0] & buf[1] & buf[2];
                        if (poss.empty() && !buf[0]) {
                            ok = 0;
                        }
                        if (ok) {
                            poss.push_back(static_cast<uint32_t>(i));
                        }
                    } else if (i + 2 == std::ssize(idx[i_mer]) - 1) {
                        if (dists[i_mer][i + 1] == d || dists[i_mer][i + 2] == d) {
                            poss.push_back(static_cast<uint32_t>(i));
                            ok = 1;
                        } else {
                            ok = 0;
                        }
                    } else if (i + 1 == std::ssize(idx[i_mer]) - 1) {
                        if (dists[i_mer][i + 1] == d) {
                            poss.push_back(static_cast<uint32_t>(i));
                            poss.push_back(static_cast<uint32_t>(i) + 1);
                            ok = 1;
                            stop = 1;
                        } else if (dists[i_mer][i] == d) {
                            poss.push_back(static_cast<uint32_t>(i));
                            ok = 1;
                            stop = 1;
                        } else {
                            ok = 0;
                            stop = 1;
                        }
                    } else {
                        spdlog::error("[kdys::{}] TRF kmer chain made impossible comparison.",
                                      __func__);
                        failed = true;
                        break;
                    }
                    if ((!ok) || stop) {
                        while (!poss.empty() && dists[i_mer][poss.back()] != d) {
                            poss.pop_back();
                        }
                        if (!poss.empty()) {
                            int start = idx[i_mer][poss[0]];
                            int end = idx[i_mer][poss.empty() ? 0 : poss.back()];
                            if (end - start > K + 2) {
                                if (start > TRF_ADD_PADDING + 1) {
                                    start -= K + TRF_ADD_PADDING + 1;
                                }
                                end += 2 * K + TRF_ADD_PADDING;
                                intervals_buf.push_back(start << 1);
                                intervals_buf.push_back(end << 1 | 1);
                                if constexpr (DEBUG_LOCAL_HAPLOTAGGING) {
                                    LOG_TRACE("[kdys::{}] d={} mer={} saw start-end: {}-{}",
                                              __func__, d, i_mer, start, end);
                                }
                                mer_is_used = 1;
                            }
                        }
                        poss.clear();
                    }
                    if (stop) {
                        break;
                    }
                }
                used[i_mer] |= mer_is_used;
                if (failed) {
                    break;
                }
            }
            if (failed) {
                break;
            }
        }

        // merge intervals (requires a minimum depth)
        if (failed /*impossible seen when parsing mers*/
            || intervals_buf.empty()) {
            if constexpr (DEBUG_LOCAL_HAPLOTAGGING) {
                LOG_TRACE(
                        "[{}] failed parsing mers (failed={}) or intervals_buf is empty "
                        "({})",
                        __func__, failed ? "true" : "false", intervals_buf.size());
            }
            intervals.clear();
        } else {
            std::sort(intervals_buf.begin(), intervals_buf.end());
            int depth = 0;
            int start = -1;
            int end = -1;
            int prev_pos = -1;
            for (size_t i = 0; i < intervals_buf.size(); i++) {
                if (intervals_buf[i] & 1) {
                    if (depth > 0) {
                        depth--;
                    }
                } else {
                    depth++;
                }

                int current_pos = intervals_buf[i] >> 1;
                const int lbreak = 0;

                if constexpr (DEBUG_LOCAL_HAPLOTAGGING) {
                    LOG_TRACE("[kdys::{}] current_pos={} stat={} depth={} lbreak={}", __func__,
                              current_pos, intervals_buf[i] & 1, depth, lbreak);
                }

                if (depth >= TRF_MIN_TANDAM_DEPTH && !lbreak) {
                    if (start == -1) {
                        start = current_pos;
                    }
                } else {
                    if (start != -1) {  // done with an interval of enough coverage, push and reset
                        end = lbreak ? prev_pos : current_pos;
                        if (end > start) {
                            // merge small gaps
                            if (!intervals.empty()) {
                                if (start - static_cast<uint32_t>(intervals.back()) <
                                    TRF_CLOSE_GAP_THRESHOLD) {  // merge block
                                    intervals.back() = (intervals.back() >> 32) << 32;
                                    intervals.back() |= end;
                                } else {  // new block
                                    intervals.push_back((static_cast<uint64_t>(start)) << 32 | end);
                                }
                            } else {  // new block
                                intervals.push_back((static_cast<uint64_t>(start)) << 32 | end);
                            }
                        }
                        start = lbreak ? current_pos : -1;
                        end = -1;
                    }
                }
                if (lbreak) {
                    depth = 0;
                }
                prev_pos = intervals_buf[i] >> 1;
            }
            if (start != -1) {
                end = intervals_buf.back() >> 1;
                if (end > start) {
                    // merge small gaps
                    if (!intervals.empty()) {
                        if (start - static_cast<uint32_t>(intervals.back()) <
                            TRF_CLOSE_GAP_THRESHOLD) {  // merge block
                            intervals.back() = (intervals.back() >> 32) << 32;
                            intervals.back() |= end;
                        } else {  // new block
                            intervals.push_back(((uint64_t)start) << 32 | end);
                        }
                    } else {  // new block
                        intervals.push_back(((uint64_t)start) << 32 | end);
                    }
                }
            }
            if constexpr (DEBUG_LOCAL_HAPLOTAGGING) {
                LOG_DEBUG("[kdys::{}] had {} intervals", __func__, intervals.size());
                for (size_t i = 0; i < intervals.size(); i++) {
                    LOG_DEBUG("[kdys::{}]    itvl#{} {}-{}", __func__, i, intervals[i] >> 32,
                              intervals[i] & std::numeric_limits<uint32_t>::max());
                }
            }
        }
    }

    return intervals;
}

std::vector<uint64_t> get_lowcmp_mask(const faidx_t *fai,
                                      const std::string_view ref_name,
                                      const int ref_start,
                                      const int ref_end) {
    const std::string span_str = create_region_string(ref_name, ref_start, ref_end);
    const std::string refseq_str = kadayashi::hts_utils::fetch_seq(fai, span_str.c_str());
    if (refseq_str.empty()) {
        return {};
    }
    return TRF_heuristic(refseq_str, ref_start);
}

bool is_lowcmp_masked(const std::vector<uint64_t> &lowcmp_intervals, const uint32_t pos0) {
    if (lowcmp_intervals.empty()) {
        return false;
    }
    const uint64_t pos = ((uint64_t)pos0) << 32 | pos0;
    const auto it = std::lower_bound(lowcmp_intervals.begin(), lowcmp_intervals.end(), pos);
    if (it != lowcmp_intervals.cbegin()) {
        const uint32_t e = static_cast<uint32_t>(*(it - 1));
        if (e > pos0) {
            return true;
        }
    }
    return false;
}

struct interval_t {
    std::string refname;
    uint32_t start;
    uint32_t end;
};
interval_t expand_query_interval(BamFileView &hf,
                                 std::string_view refname,
                                 const uint32_t itvl_start,
                                 const uint32_t itvl_end) {
    // Adjust region start and end: if there happens to be no
    // heterozygous variants around start and/or end, we could have
    // 30-50% unread unphased. Here, we allow expanding the
    // target region *somewhat*: the start/end of the first/last
    // read, if they are not too long; or +-50kb when they are
    // too long.
    constexpr uint32_t OFFSET_DEFAULT = 50000;

    const std::string itvl_left =
            create_region_string(refname, itvl_start, itvl_start + 1);  // +1 to get 1 base

    BamPtr aln = BamPtr(bam_init1(), BamDestructor());
    HtsItrPtr bamitr =
            HtsItrPtr(sam_itr_querys(hf.idx, hf.hdr, itvl_left.c_str()), HtsItrDestructor());

    uint32_t offset_left = OFFSET_DEFAULT;
    int offset_right = 0;

    // left
    while (sam_itr_next(hf.fp, bamitr.get(), aln.get()) >= 0) {
        if ((itvl_start < OFFSET_DEFAULT + aln->core.pos) && (itvl_start > aln->core.pos)) {
            offset_left = itvl_start - static_cast<uint32_t>(aln->core.pos);
            break;
        }
    }

    //right
    const std::string itvl_right =
            create_region_string(refname, itvl_end, itvl_end + 1);  // +1 to get 1 base
    bamitr = HtsItrPtr(sam_itr_querys(hf.idx, hf.hdr, itvl_right.c_str()), HtsItrDestructor());
    while (sam_itr_next(hf.fp, bamitr.get(), aln.get()) >= 0) {
        const uint32_t end_pos = static_cast<uint32_t>(bam_endpos(aln.get()));
        if (end_pos < OFFSET_DEFAULT + itvl_end) {
            offset_right = offset_right + itvl_end > end_pos ? offset_right : end_pos - itvl_end;
        }
    }
    if (offset_right == 0) {
        offset_right = OFFSET_DEFAULT;
    }

    // finalize
    const uint32_t abs_start = itvl_start <= offset_left ? 0 : itvl_start - offset_left;
    const uint32_t abs_end = itvl_end + offset_right;

    interval_t ret = {.refname = std::string(refname), .start = abs_start, .end = abs_end};
    return ret;
}

bool is_adjacent_to_perfect_repeats1(const char *seq,
                                     const uint32_t seq_l,
                                     const uint32_t pattern_l) {
    // check full length of seq for perfect hp/dimer/trimer
    assert(pattern_l < seq_l);
    for (uint32_t i = 0; i + pattern_l < seq_l; i++) {
        if (seq[i] != seq[i + pattern_l]) {
            return false;
        }
    }
    return true;
}

bool is_adjacent_to_perfect_repeats(const char *refseq,
                                    const uint32_t refseq_l,
                                    const uint32_t refseq_start,
                                    const uint32_t pos) {
    // check perfect hp around spot, dimer or trimer; for other use low complexity mask
    if (refseq_l < 10) {
        return false;
    }
    if (refseq_start > pos) {
        spdlog::error("[kdys::{}] shouldn't happen, bad input check code (pos {} refseq_start {})",
                      __func__, pos, refseq_start);
        return false;
    }

    constexpr int LEN = 4;
    const std::array<uint32_t, LEN> spans = {6, 6, 9, 12};
    const std::array<uint32_t, LEN> pattern_ls = {1, 2, 3, 4};
    for (uint32_t is = 0; is < LEN; is++) {
        const uint32_t span = spans[is];
        const uint32_t pattern_l = pattern_ls[is];

        uint32_t left_start = pos - refseq_start;
        left_start = left_start < span ? 0 : left_start - span;

        const std::array<uint32_t, LEN> starts = {left_start, left_start + 1, pos - refseq_start,
                                                  pos - refseq_start + 1};
        for (int i = 0; i < LEN; i++) {
            if (starts[i] + span >= refseq_l) {
                continue;
            }
            const bool is_perfect_repeat =
                    is_adjacent_to_perfect_repeats1(refseq + starts[i], span, pattern_l);
            if (is_perfect_repeat) {
                return true;
            }
        }
    }
    return false;
}

bool classify_variant_prefilter(vc_variants1_val_t &var,
                                const uint32_t pos,
                                const char *refseq,
                                const int refseq_l,
                                const uint32_t itvl_start,
                                const str2int_t *qname2hp) {
    // return true if variant is being classified & shall not proceed
    // return false otherwise

    constexpr int MIN_COV_ONE = 3;
    constexpr int MIN_COV_TOT = 5;
    constexpr int LOW_COV = 15;
    constexpr int LOW_COV2 = 20;
    constexpr int SMALL_COV_DIFF = 3;
    constexpr float MIN_ALT_FRAC = 0.2f;
    constexpr float MAX_UNPHASED_FRAC = 0.7f;
    constexpr float MIN_HOMALT_FRAC = 0.9f;

    if (var.is_accepted != FLAG_VARSTAT_MAYBE) {
        // ref allele was not collected for the candidate,
        // only consider rescues
        float tot_cov_alt = 0;
        float tot_cov_any = 0;
        int suf_alt = 0;
        for (auto allele : var.alleles) {
            int tmp = allele.cov.cov_hap0 + allele.cov.cov_hap1 + allele.cov.cov_unphased;
            tot_cov_any += static_cast<float>(tmp);
            if (allele.allele[0] != SENTINEL_REF_ALLELE_INT) {
                tot_cov_alt += static_cast<float>(tmp);
                if (tmp > 1) {
                    suf_alt++;
                }
            }
        }

        if (qname2hp && var.alleles.size() >= 2 && tot_cov_alt > MIN_COV_TOT &&
            suf_alt >= MIN_COV_ONE && tot_cov_alt / tot_cov_any > MIN_ALT_FRAC) {
            if constexpr (DEBUG_LOCAL_HAPLOTAGGING) {
                LOG_DEBUG(
                        "[{}] let pos {} set to unsure although flag is bad, due to its "
                        "many alleles & sufficient coverage in alt ({}/{})",
                        __func__, pos, tot_cov_alt, tot_cov_any);
            }
            var.is_accepted = FLAG_VARSTAT_UNSURE;
            var.type = FLAG_VAR_NA;
            return true;
        } else {
            if constexpr (DEBUG_LOCAL_HAPLOTAGGING) {
                LOG_DEBUG(
                        "[{}] straight ignore pos {} (flag is bad; size is {}, tot cov is "
                        "{})",
                        __func__, pos, var.alleles.size(), tot_cov_alt);
            }
            var.is_accepted = FLAG_VARSTAT_REJECTED;
            var.type = FLAG_VAR_NA;
            return true;
        }
    } else {
        // default to reject
        var.is_accepted = FLAG_VARSTAT_REJECTED;

        // should not happen
        if (var.alleles.size() < 3) {  // 3: alt *1, ref placeholders *2
            spdlog::error("[kdys::{}] bad number of alleles (var.alleles.size={})", __func__,
                          var.alleles.size());
            return true;  // not really done, but return true to let caller ignore this candidate
        }

        // (make coverage easier to access)
        var.type = FLAG_VAR_HET;
        vc_allele_t &a0 = var.alleles[0];
        vc_allele_t &a1 = var.alleles[1];
        const float top_cov_1 = static_cast<float>(qname2hp ? var.alleles[0].cov.cov_tot_phased
                                                            : var.alleles[0].cov.cov_unphased);
        const float top_cov_2 = static_cast<float>(qname2hp ? var.alleles[1].cov.cov_tot_phased
                                                            : var.alleles[1].cov.cov_unphased);
        const float top_cov_1_unphased = static_cast<float>(var.alleles[0].cov.cov_unphased);
        const float top_cov_2_unphased = static_cast<float>(var.alleles[1].cov.cov_unphased);
        const float top_two_cov = top_cov_1 + top_cov_2;
        const float tot_cov_unphased = top_cov_1_unphased + top_cov_2_unphased;
        float tot_cov = top_two_cov;
        float tot_all_non_refs = 0;
        for (uint32_t i = 2; i < var.alleles.size(); i++) {
            tot_cov += static_cast<float>(qname2hp ? var.alleles[i].cov.cov_tot_phased
                                                   : var.alleles[i].cov.cov_unphased);
        }
        for (uint32_t i = 0; i < var.alleles.size(); i++) {
            if (!(var.alleles[i].allele[0] == SENTINEL_REF_ALLELE_INT &&
                  var.alleles[i].allele.back() == VAR_OP_X)) {
                tot_all_non_refs += static_cast<float>(var.alleles[i].cov.cov_hap0) +
                                    static_cast<float>(var.alleles[i].cov.cov_hap1) +
                                    static_cast<float>(var.alleles[i].cov.cov_unphased);
            }
        }

        // first 2 are both REF sentinel
        if (var.alleles[0].allele[0] == SENTINEL_REF_ALLELE_INT &&
            var.alleles[1].allele[0] == SENTINEL_REF_ALLELE_INT) {
            var.is_accepted = FLAG_VARSTAT_REJECTED;
            var.type = FLAG_VAR_NA;
            if constexpr (DEBUG_LOCAL_HAPLOTAGGING) {
                LOG_DEBUG("[kdys::{}] ignore pos {} (first two are REF sentinels)", __func__, pos);
            }
            return true;
        }

        // if we are doing phased pileup,
        // mark candiate with substantial unphased coverage as unsure
        if (qname2hp) {
            const float top_cov_12_unphased = top_cov_1_unphased + top_cov_2_unphased;
            const int has_many_alts1 =
                    (tot_cov + tot_cov_unphased >= LOW_COV2 &&
                     tot_all_non_refs / (tot_cov + tot_cov_unphased) >= MIN_ALT_FRAC);
            const int has_many_alts2 =
                    (tot_cov + tot_cov_unphased < LOW_COV2 && tot_all_non_refs >= MIN_COV_ONE);
            if (top_cov_12_unphased > (top_cov_1 + top_cov_2) &&
                (has_many_alts1 || has_many_alts2)) {
                // alt has enough coverage; we will only consider hom case here
                if (top_cov_1_unphased / tot_cov_unphased >= MAX_UNPHASED_FRAC &&
                    tot_cov_unphased >= LOW_COV2) {
                    var.is_accepted = FLAG_VARSTAT_UNSURE;
                    var.type = FLAG_VAR_HOM;
                    if constexpr (DEBUG_LOCAL_HAPLOTAGGING) {
                        LOG_DEBUG(
                                "[{}] set pos {} as unsure (due to unphased) tot_all_non_refs={} "
                                "totcov={}",
                                __func__, pos, tot_all_non_refs, tot_cov);
                    }
                    return true;
                }
            }
        }

        // caution: although this prefilter only awards hom to accepted candidates,
        // we need to check phasing (if is phased pileup), as strong hap-specific
        // coverage can be sever as 12 alts in hap0, 2 ref in hap1 and nothing else.
        double fisher_left_p_nounphased, fisher_right_p_nounphased, fisher_twosided_p_nounphased;
        kt_fisher_exact(a0.cov.cov_hap0, a0.cov.cov_hap1, a1.cov.cov_hap0, a1.cov.cov_hap1,
                        &fisher_left_p_nounphased, &fisher_right_p_nounphased,
                        &fisher_twosided_p_nounphased);
        if constexpr (DEBUG_LOCAL_HAPLOTAGGING) {
            LOG_DEBUG(
                    "[{}] pos {} in prefiltering fisher: phased-only, twosided p: {:.4f} ({} "
                    "{}, {} {})",
                    __func__, pos, fisher_twosided_p_nounphased, a0.cov.cov_hap0, a0.cov.cov_hap1,
                    a1.cov.cov_hap0, a1.cov.cov_hap1);
        }

        // strong hom case
        int is_hom = 0;
        if (var.alleles[0].allele[0] != SENTINEL_REF_ALLELE_INT) {
            if (!qname2hp) {
                if (top_cov_1 / tot_cov >= 0.7f && tot_cov >= LOW_COV2) {
                    is_hom = 1;
                    if constexpr (DEBUG_LOCAL_HAPLOTAGGING) {
                        LOG_DEBUG("[kdys::{}] set pos {} as hom (2, unphased; {}/{})", __func__,
                                  pos, top_cov_1, tot_cov);
                    }
                }
            } else {
                const float hap0_tot = std::accumulate(var.alleles.begin(), var.alleles.end(), 0.0f,
                                                       [](float acc, const auto &allele) -> float {
                                                           return acc + (float)allele.cov.cov_hap0;
                                                       });
                const float hap1_tot = std::accumulate(var.alleles.begin(), var.alleles.end(), 0.0f,
                                                       [](float acc, const auto &allele) -> float {
                                                           return acc + (float)allele.cov.cov_hap1;
                                                       });
                const float tot_cov2 = std::accumulate(var.alleles.begin(), var.alleles.end(), 0.0f,
                                                       [](float acc, const auto &allele) -> float {
                                                           return acc + (float)allele.cov.cov_hap0 +
                                                                  (float)allele.cov.cov_hap1 +
                                                                  (float)allele.cov.cov_unphased;
                                                       });
                const float hap0 = static_cast<float>(var.alleles[0].cov.cov_hap0);
                const float hap1 = static_cast<float>(var.alleles[0].cov.cov_hap1);
                const float hap_unphased = static_cast<float>(var.alleles[0].cov.cov_unphased);
                const float hap_all = hap0 + hap1 + hap_unphased;
                const float threshold = var.alleles[0].allele.back() == VAR_OP_X ? 0.7f : 0.5f;
                if (hap0_tot > MIN_COV_ONE && hap1_tot > MIN_COV_ONE &&
                    hap0 / hap0_tot >= threshold && hap1 / hap1_tot >= threshold) {
                    is_hom = 1;
                    if constexpr (DEBUG_LOCAL_HAPLOTAGGING) {
                        LOG_DEBUG("[kdys::{}] set pos {} as hom (2, phased; {}/{}, {}/{})",
                                  __func__, pos, hap0, hap0_tot, hap1, hap1_tot);
                    }
                } else if (
                        hap_all / tot_cov2 >= MIN_HOMALT_FRAC ||
                        (tot_cov2 - hap_all < SMALL_COV_DIFF &&
                         hap_all >
                                 LOW_COV)) {  // alternatively, as long as we have almost all bases being the one alt allele, call hom regardless of phasing quality
                    is_hom = 1;
                    if constexpr (DEBUG_LOCAL_HAPLOTAGGING) {
                        LOG_DEBUG("[kdys::{}] set pos {} as hom (2b; hap_all={} tot_cov2={})",
                                  __func__, pos, hap_all, tot_cov2);
                    }
                } else if (hap_all / tot_cov2 > 0.8f &&
                           (int)(tot_cov2 - tot_all_non_refs) < SMALL_COV_DIFF) {
                    is_hom = 1;
                    if constexpr (DEBUG_LOCAL_HAPLOTAGGING) {
                        LOG_DEBUG(
                                "[{}] set pos {} as hom (2c; allel0all={}, tot_cov2={}, "
                                "totallnonref={})",
                                __func__, pos, hap_all, tot_cov2, tot_all_non_refs);
                    }
                }
            }
        }

        const bool is_at_repeat = is_adjacent_to_perfect_repeats(refseq, refseq_l, itvl_start, pos);
        if (is_hom) {
            var.is_accepted = FLAG_VARSTAT_ACCEPTED;
            var.type = FLAG_VAR_HOM;
            if (is_at_repeat || fisher_twosided_p_nounphased < 0.05f) {
                var.is_accepted = FLAG_VARSTAT_UNSURE;
                if constexpr (DEBUG_LOCAL_HAPLOTAGGING) {
                    LOG_DEBUG(
                            "[{}] pos {} was set to accepted hom, but flipping this to unsure "
                            "due to nearby repeat; type flag is now {}",
                            __func__, pos, var.type);
                }
            }
            return true;
        }
        if (top_two_cov / tot_cov <
            0.8f) {  // top two calls did not dominate over all observations (e.g. INS at a homopolyer)
            if (std::min(top_cov_1, top_cov_2) >= 5 || tot_all_non_refs >= 10 || is_at_repeat) {
                var.is_accepted = FLAG_VARSTAT_UNSURE;
                if constexpr (DEBUG_LOCAL_HAPLOTAGGING) {
                    LOG_DEBUG(
                            "[{}] set pos {} unsure (1a: weird coverage (top1={} top2={}; all "
                            "alts={}))",
                            __func__, pos, top_cov_1, top_cov_2, tot_all_non_refs);
                }
            } else {
                var.is_accepted = FLAG_VARSTAT_REJECTED;
                if constexpr (DEBUG_LOCAL_HAPLOTAGGING) {
                    LOG_DEBUG(
                            "[{}] reject pos {} (1a: weird coverage ({} alleles) and they are "
                            "low: {} (base_int={}) and {} (base_int={}) while top2 cov={}, tot cov "
                            "is {} )",
                            __func__, pos, var.alleles.size(), top_cov_1, var.alleles[0].allele[0],
                            top_cov_2, var.alleles[1].allele[0], top_two_cov, tot_cov);
                }
            }
            return true;
        }

        if constexpr (DEBUG_LOCAL_HAPLOTAGGING) {
            if (qname2hp) {
                LOG_DEBUG(
                        "[kdys::{}] pos {} remained, two top covs: phased=({} and {}) unphased=({} "
                        "and "
                        "{})",
                        __func__, pos, top_cov_1, top_cov_2, top_cov_1_unphased,
                        top_cov_2_unphased);
            }
        }

        return false;
    }
}

enum var_classify_code {
    UNSET,

    // check het first
    HET_FISHER_COV_STRAND,  // "het, passed fisher test, passed coverage eveness requirement, passed strand no-bias requirement"
    HET_FISHER,
    HET_WEAKERFISHER,
    HET_NOTHING,
    REJECT_TOO_MANY_ALLELE,
    REJECT_REFREF,

    // then consider hom
    HOM_STRAND,  // "hom, passed strand biase requirement"
    HOM_NOTHING,
    HOM_RESUCEFAIL,

    // we may still set unsure before returning
    SUS_LONG_ALT,
    SUS_MULTI_ALLELE,
    SUS_REPETITIVE,
    SUS_PHASE
};
struct var_classify_t {
    uint8_t type;
    uint8_t is_accepted;
    var_classify_code code;
    std::string info;
};
struct cov_extended_t {
    cov_t cov;
    int cov_any_alt = 0;  // both phased and unphased
    int n_long_alts = 0;
    int n_alts_in_first_two = 0;
    const int min_len_long_alt = 5;
};
void variant_fill_cov_tot(vc_variants1_val_t &var) {
    for (auto &allele : var.alleles) {
        allele.cov.cov_tot_phased = allele.cov.cov_hap0 + allele.cov.cov_hap1;
        allele.cov.cov_tot_all = allele.cov.cov_tot_phased + allele.cov.cov_unphased;
    }
}
cov_extended_t variant_get_all_allele_coverage(const vc_variants1_val_t &var) {
    cov_extended_t ret{};

    for (const auto &allele : var.alleles) {
        ret.cov.cov_hap0 += allele.cov.cov_hap0;
        ret.cov.cov_hap1 += allele.cov.cov_hap1;
        ret.cov.cov_unphased += allele.cov.cov_unphased;

        ret.cov.cov_tot_all += allele.cov.cov_tot_all;
        ret.cov.cov_tot_phased += allele.cov.cov_tot_phased;

        ret.cov.cov_fwd += allele.cov.cov_fwd;
        ret.cov.cov_bwd += allele.cov.cov_bwd;

        if (allele.allele[0] != SENTINEL_REF_ALLELE_INT) {
            ret.cov_any_alt += allele.cov.cov_tot_all;
        }

        if (allele.allele[0] != SENTINEL_REF_ALLELE_INT &&
            static_cast<int>(allele.allele.size()) >= ret.min_len_long_alt) {
            ret.n_long_alts += allele.cov.cov_tot_all;
        }
    }
    if (var.alleles[0].allele[0] != SENTINEL_REF_ALLELE_INT) {
        ret.n_alts_in_first_two++;
    }
    if (var.alleles[1].allele[0] != SENTINEL_REF_ALLELE_INT) {
        ret.n_alts_in_first_two++;
    }
    return ret;
}

var_classify_t classify_variant_phased(vc_variants1_val_t &var,
                                       const uint32_t pos,
                                       const char *refseq,
                                       const int refseq_l,
                                       const uint32_t itvl_start,
                                       const int min_strand_cov,
                                       const float min_strand_cov_frac) {
    // Candidate should have its ref allele collected.
    assert(var.alleles.size() > 1);

    var_classify_t ret;
    ret.is_accepted = FLAG_VARSTAT_REJECTED;
    ret.type = FLAG_VAR_NA;
    ret.code = UNSET;

    constexpr int N_LONG_ALTS = 3;
    constexpr int MIN_SUF_COV_ALLELE = 3;

    constexpr float UNPHASED_DISCOUNT = 2.0f;
    constexpr float FISHER_P_THRESHOLD = 0.005f;
    constexpr float FISHER_P_THRESHOLD_NOUNPHASED = 0.01f;

    // prep coverage
    cov_extended_t varcov = variant_get_all_allele_coverage(var);
    vc_allele_t &a0 = var.alleles[0];
    vc_allele_t &a1 = var.alleles[1];

    // regular het requirements
    // (cov diff)
    constexpr int MAX_COVDIFF_RATIO = 3;
    constexpr int MIN_HAP_COVERAGE = 5;
    auto has_cov_diff_between_two_alleles = [](const vc_allele_t &allele_A,
                                               const vc_allele_t &allele_B) -> bool {
        const int hi = std::max(allele_A.cov.cov_tot_all, allele_B.cov.cov_tot_all);
        const int lo = std::min(allele_A.cov.cov_tot_all, allele_B.cov.cov_tot_all);
        return ((static_cast<float>(hi) / static_cast<float>(lo)) >= MAX_COVDIFF_RATIO) ||
               lo < MIN_HAP_COVERAGE;
    };
    // (intra-haplotype ratio)
    constexpr float MIN_INHAP_RATIO = 0.8f;
    // note: for min_strand_cov, >=2 is too much for 30x in regions prone to coverage drop,
    // but 1 may be too low and allows confident false positives.
    constexpr int LOWCOV_THRESHOLD = 15;
    constexpr int LOWCOV_ALT_THRESHOLD = 3;
    enum which_hap { HAP0, HAP1 };
    auto allele_dominates_haplotype = [&varcov](const vc_allele_t &allele, which_hap hap) {
        int a_cov_hap;
        int tot_cov_hap;
        if (hap == HAP0) {
            a_cov_hap = allele.cov.cov_hap0;
            tot_cov_hap = varcov.cov.cov_hap0;
        } else {
            a_cov_hap = allele.cov.cov_hap1;
            tot_cov_hap = varcov.cov.cov_hap1;
        }
        const bool lowcov_ok = tot_cov_hap < LOWCOV_THRESHOLD && a_cov_hap < LOWCOV_ALT_THRESHOLD;
        const bool regular_ok =
                a_cov_hap < static_cast<int>(MIN_INHAP_RATIO * static_cast<float>(tot_cov_hap));
        return regular_ok || lowcov_ok;
    };
    auto allele_is_cleanly_phased = [&allele_dominates_haplotype](vc_allele_t &a) {
        const bool dominates_hap0 =
                allele_dominates_haplotype(a, HAP0) && !allele_dominates_haplotype(a, HAP1);
        const bool dominates_hap1 =
                allele_dominates_haplotype(a, HAP1) && !allele_dominates_haplotype(a, HAP0);
        return dominates_hap0 || dominates_hap1;
    };

    // weaker het requirements
    constexpr float WEAKTHRESHOLD_MIN_ALT_RATIO = 0.3f;

    // hom
    constexpr float THRESHOLD_MIN_HOM_RATIO = 0.8f;

    double fisher_left_p, fisher_right_p, fisher_twosided_p;
    double fisher_left_p_nounphased, fisher_right_p_nounphased, fisher_twosided_p_nounphased;
    kt_fisher_exact(a0.cov.cov_hap0 + a0.cov.cov_unphased / static_cast<int>(UNPHASED_DISCOUNT),
                    a0.cov.cov_hap1 + a0.cov.cov_unphased / static_cast<int>(UNPHASED_DISCOUNT),
                    a1.cov.cov_hap0 + a1.cov.cov_unphased / static_cast<int>(UNPHASED_DISCOUNT),
                    a1.cov.cov_hap1 + a1.cov.cov_unphased / static_cast<int>(UNPHASED_DISCOUNT),
                    &fisher_left_p, &fisher_right_p, &fisher_twosided_p);
    kt_fisher_exact(a0.cov.cov_hap0, a0.cov.cov_hap1, a1.cov.cov_hap0, a1.cov.cov_hap1,
                    &fisher_left_p_nounphased, &fisher_right_p_nounphased,
                    &fisher_twosided_p_nounphased);

    bool has_decided = false;

    // het
    const bool passed_fisher = fisher_twosided_p < FISHER_P_THRESHOLD ||
                               (fisher_twosided_p_nounphased < FISHER_P_THRESHOLD_NOUNPHASED &&
                                varcov.n_alts_in_first_two > 0);
    const bool has_cov_diff = has_cov_diff_between_two_alleles(a0, a1);
    const bool alt0_is_proper = allele_is_cleanly_phased(a0);
    const bool alt1_is_proper = allele_is_cleanly_phased(a1);

    auto strand_has_enough_cov = [&min_strand_cov, &min_strand_cov_frac](const cov_t &cov) -> bool {
        return std::min(cov.cov_fwd, cov.cov_bwd) >=
               std::max(min_strand_cov, static_cast<int>(static_cast<float>(cov.cov_tot_all) *
                                                         min_strand_cov_frac));
    };
    const bool alt0_strand_ok = strand_has_enough_cov(a0.cov);
    const bool alt1_strand_ok = strand_has_enough_cov(a1.cov);

    const bool ok = passed_fisher && alt0_is_proper && alt1_is_proper;
    const bool ok_weaker =
            (varcov.cov_any_alt >=
             WEAKTHRESHOLD_MIN_ALT_RATIO * static_cast<float>(varcov.cov.cov_tot_all)) ||
            passed_fisher;
    if (ok) {
        if (!has_cov_diff && alt0_strand_ok && alt1_strand_ok) {
            ret.is_accepted = FLAG_VARSTAT_ACCEPTED;
            ret.code = HET_FISHER_COV_STRAND;
        } else {
            ret.is_accepted = FLAG_VARSTAT_UNSURE;
            ret.code = HET_FISHER;
        }
    } else {
        if (ok_weaker) {
            ret.is_accepted = FLAG_VARSTAT_UNSURE;
            ret.code = HET_WEAKERFISHER;
        } else {
            ret.is_accepted = FLAG_VARSTAT_REJECTED;
            ret.code = HET_NOTHING;
        }
    }

    if (ret.is_accepted !=
        FLAG_VARSTAT_REJECTED) {  // is het, need to decide whether it has multi alleles
        if (varcov.n_alts_in_first_two == 2) {
            ret.type = FLAG_VAR_MULTHET;
        } else if (varcov.n_alts_in_first_two == 1) {
            ret.type = FLAG_VAR_HET;
        } else {
            ret.is_accepted = FLAG_VARSTAT_REJECTED;
            ret.type = FLAG_VAR_NA;
            ret.code = REJECT_TOO_MANY_ALLELE;
        }
        has_decided = true;
    }

    // not clean het, and top two are ref(substitution) and ref(del), ignore
    if (!has_decided && var.alleles[0].allele[0] == SENTINEL_REF_ALLELE_INT &&
        var.alleles[1].allele[0] == SENTINEL_REF_ALLELE_INT) {
        ret.is_accepted = FLAG_VARSTAT_REJECTED;
        ret.type = FLAG_VAR_NA;
        ret.code = REJECT_REFREF;
        has_decided = true;
    }

    // hom
    if (!has_decided && var.alleles[0].allele[0] != SENTINEL_REF_ALLELE_INT) {
        const float ratio =
                (float)a0.cov.cov_tot_all / static_cast<float>(std::max(1, varcov.cov.cov_tot_all));
        if (ratio >= THRESHOLD_MIN_HOM_RATIO) {
            ret.type = FLAG_VAR_HOM;
            if (alt0_strand_ok) {
                ret.is_accepted = FLAG_VARSTAT_ACCEPTED;
                ret.code = HOM_STRAND;
            } else {
                ret.is_accepted = FLAG_VARSTAT_UNSURE;
                ret.code = HOM_NOTHING;
            }
            has_decided = true;
        }
    }
    if (!has_decided) {
        for (auto &allele : var.alleles) {
            if (allele.allele[0] != SENTINEL_REF_ALLELE_INT &&
                static_cast<float>(allele.cov.cov_tot_all) >= 0.3f * varcov.cov.cov_tot_all) {
                ret.is_accepted = FLAG_VARSTAT_UNSURE;
                ret.type = FLAG_VAR_NA;
                ret.code = HOM_RESUCEFAIL;
                has_decided = true;
                break;
            }
        }
    }

    if (!has_decided) {
        if (varcov.n_long_alts >= N_LONG_ALTS) {  // long alt
            ret.is_accepted = FLAG_VARSTAT_UNSURE;
            ret.code = SUS_LONG_ALT;
        } else if (var.alleles.size() > 4) {  // many alts; 4:ref/ref/alt/alt
            int n_allele_with_suf_cov = 0;
            for (auto &allele : var.alleles) {
                if (allele.allele[0] != SENTINEL_REF_ALLELE_INT &&
                    allele.cov.cov_tot_all >= MIN_SUF_COV_ALLELE) {
                    n_allele_with_suf_cov++;
                }
            }
            if (n_allele_with_suf_cov >= 2) {
                ret.is_accepted = FLAG_VARSTAT_UNSURE;
                ret.code = SUS_MULTI_ALLELE;
            }
        }
    }

    // additional criteria for marking candidate as unsure at the last moment
    // (1) rescue candidates close to repeats
    // (2) untrust indel candidates if close repeats
    const bool is_non_ref =
            var.alleles.size() > 1 &&
            static_cast<float>(varcov.cov_any_alt) >=
                    (0.15f * static_cast<float>(varcov.cov.cov_tot_all)) &&
            (!(a0.allele.back() == VAR_OP_X && a0.allele[0] == SENTINEL_REF_ALLELE_INT) ||
             !(a1.allele.back() != VAR_OP_X && a1.allele[0] == SENTINEL_REF_ALLELE_INT));
    const int is_rejected_non_ref = is_non_ref && var.is_accepted == FLAG_VARSTAT_REJECTED;
    int is_indel = 0;
    int n_long_indel = 0;
    if (var.alleles.size() > 1) {
        is_indel = (a0.allele.back() == VAR_OP_I || a1.allele.back() == VAR_OP_I ||
                    a0.allele.back() == VAR_OP_D || a1.allele.back() == VAR_OP_D);
        for (uint32_t i = 0; i < var.alleles.size(); i++) {
            if (var.alleles[i].allele[0] != SENTINEL_REF_ALLELE_INT &&
                var.alleles[i].allele.size() > 10) {
                n_long_indel += var.alleles[i].cov.cov_hap0 + var.alleles[i].cov.cov_hap1 +
                                var.alleles[i].cov.cov_unphased;
            }
        }
    }

    bool is_at_repeat = false;
    if (is_rejected_non_ref || is_indel || n_long_indel > 1) {
        if (static_cast<float>(varcov.cov_any_alt) >=
            0.2f * static_cast<float>(varcov.cov.cov_tot_all)) {
            is_at_repeat = is_adjacent_to_perfect_repeats(refseq, refseq_l, itvl_start, pos);
            if (is_at_repeat) {
                ret.is_accepted = FLAG_VARSTAT_UNSURE;
                ret.code = SUS_REPETITIVE;
            }
        }
    }
    if (ret.is_accepted == FLAG_VARSTAT_ACCEPTED &&
        (varcov.cov.cov_tot_all - varcov.cov.cov_tot_phased) >
                (varcov.cov.cov_hap0 + varcov.cov.cov_hap1)) {
        ret.is_accepted = FLAG_VARSTAT_UNSURE;
        ret.code = SUS_PHASE;
    }

    if constexpr (DEBUG_LOCAL_HAPLOTAGGING) {
        LOG_DEBUG(
                "[{}] pos={} is_accepted={} type={} code={}; fisher_p={}; is_at_repeat={}; "
                "a0({}{})=[{} {}] a1({}{})=[{} {}] "
                "proper: {} {}; strand covs: {} {} {} {}",
                __func__, pos, ret.is_accepted, ret.type, ret.code, fisher_twosided_p_nounphased,
                is_at_repeat,

                "ACGT_R"[var.alleles[0].allele[0]], var.alleles[0].allele.back(), a0.cov.cov_hap0,
                a0.cov.cov_hap1, "ACGT_R"[var.alleles[1].allele[0]], var.alleles[1].allele.back(),
                a1.cov.cov_hap0, a1.cov.cov_hap1,

                alt0_is_proper, alt1_is_proper, a0.cov.cov_fwd, a0.cov.cov_bwd, a1.cov.cov_fwd,
                a1.cov.cov_bwd);
    }

    return ret;
}

struct valid_pos_t {
    uint32_t pos;
    uint8_t stat;  // FLAG_VARSTAT_...
};

phase_return_t kadayashi_dvr_single_region_wrapper1(samFile *fp_bam,
                                                    hts_idx_t *fp_bai,
                                                    sam_hdr_t *fp_header,
                                                    const faidx_t *fai,
                                                    const std::string_view ref_name,
                                                    const uint32_t ref_start,
                                                    const uint32_t ref_end,
                                                    const bool disable_interval_expansion,
                                                    const int min_base_quality,
                                                    const int min_varcall_coverage,
                                                    const float min_varcall_fraction,
                                                    const int max_clipping,
                                                    const int min_strand_cov,
                                                    const float min_strand_cov_frac,
                                                    const float max_gapcompressed_seqdiv) {
    const pileup_pars_t pp = {
            .min_base_quality = min_base_quality,
            .min_varcall_coverage = min_varcall_coverage,
            .min_varcall_fraction = min_varcall_fraction,
            .max_clipping = max_clipping,
            .min_strand_cov = min_strand_cov,
            .min_strand_cov_frac = min_strand_cov_frac,
            .max_gapcompressed_seqdiv = max_gapcompressed_seqdiv,
            .retain_het_only = true,
            .retain_SNP_only = true,
            .use_bloomfilter = false,
            .disable_low_complexity_masking = false,
            .disable_region_expansion = static_cast<bool>(!!disable_interval_expansion)};

    return kadayashi::kadayashi_local_haptagging_dvr_single_region(
            fp_bam, fp_bai, fp_header, fai, ref_name, ref_start, ref_end, pp);
}
phase_return_t kadayashi_simple_single_region_wrapper1(samFile *fp_bam,
                                                       hts_idx_t *fp_bai,
                                                       sam_hdr_t *fp_header,
                                                       const faidx_t *fai,
                                                       const std::string_view ref_name,
                                                       const uint32_t ref_start,
                                                       const uint32_t ref_end,
                                                       const bool disable_interval_expansion,
                                                       const int min_base_quality,
                                                       const int min_varcall_coverage,
                                                       const float min_varcall_fraction,
                                                       const int max_clipping,
                                                       const int min_strand_cov,
                                                       const float min_strand_cov_frac,
                                                       const float max_gapcompressed_seqdiv) {
    const pileup_pars_t pp = {
            .min_base_quality = min_base_quality,
            .min_varcall_coverage = min_varcall_coverage,
            .min_varcall_fraction = min_varcall_fraction,
            .max_clipping = max_clipping,
            .min_strand_cov = min_strand_cov,
            .min_strand_cov_frac = min_strand_cov_frac,
            .max_gapcompressed_seqdiv = max_gapcompressed_seqdiv,
            .retain_het_only = true,
            .retain_SNP_only = true,
            .use_bloomfilter = false,
            .disable_low_complexity_masking = false,
            .disable_region_expansion = static_cast<bool>(!!disable_interval_expansion)};

    return kadayashi::kadayashi_local_haptagging_simple_single_region(
            fp_bam, fp_bai, fp_header, fai, ref_name, ref_start, ref_end, pp);
}

void ck_derive_variant_genophase_from_phased_read(chunk_t &ck) {
    constexpr int MIN_AMBIGUOUS_ALT_COV = 4;
    for (ta_t &var : ck.varcalls) {
        var.genotype = {'.', '/', '.', '\0'};

        if (var.is_used == TA_STAT_UNSURE) {
            if (var.type == TA_TYPE_HOM) {
                var.genotype[0] = '1';
                var.genotype[2] = '1';
            } else {
                var.genotype[0] = '0';
                var.genotype[2] = '1';
            }
            continue;
        }

        if (var.type == TA_TYPE_HOM) {
            var.genotype[0] = '1';
            var.genotype[2] = '1';
        } else {
            int n_hap0 = 0;
            int n_hap1 = 0;

            // default case: for multi-allele het we always find out
            //  the haptag of the first allele (as in buffer's order).
            // For regular het, figure out which in the buffer is ALT
            // and find its haptag.
            std::span<const uint32_t> readIDs{var.allele2readIDs[0]};
            if (var.type == TA_TYPE_HET) {
                if (var.alleles[0][0] == SENTINEL_REF_ALLELE_INT) {
                    readIDs = var.allele2readIDs[0];
                } else {
                    readIDs = var.allele2readIDs[1];
                }
            }

            for (size_t i = 0; i < readIDs.size(); i++) {
                const uint32_t readID = readIDs[i];
                if (ck.reads[readID].hp == 0) {
                    n_hap0++;
                } else if (ck.reads[readID].hp == 1) {
                    n_hap1++;
                }
            }
            if (n_hap0 >= MIN_AMBIGUOUS_ALT_COV && n_hap1 >= MIN_AMBIGUOUS_ALT_COV) {
                // unphased
                var.genotype[0] = '0';
                var.genotype[2] = '1';
            } else {
                // note: here set both het and multihet to 01, let this be resolved when writing vcf
                if (n_hap0 > n_hap1) {
                    var.genotype = {'0', '|', '1', '\0'};
                } else if (n_hap1 > n_hap0) {
                    var.genotype = {'1', '|', '0', '\0'};
                } else {  // unphased
                    var.genotype[0] = '0';
                    var.genotype[2] = '1';
                }
            }
        }  // determine het variant phase+geno
    }  // iter through variants
}

variant_dorado_style_t convert_fullinfo_var_to_dorado_style(const variant_fullinfo_t &var) {
    std::pair<char, char> genotype;
    std::string_view ref = var.ref_allele_seq0;
    std::string alt0 = var.alt_allele_seq0;
    std::string alt1 = var.is_multi_allele ? var.alt_allele_seq1 : "";
    if (var.is_multi_allele) {
        if (var.genotype0[0] == '0') {
            genotype = {'2', '1'};
        } else {
            genotype = {'1', '2'};
        }
        if (var.ref_allele_seq0 != var.ref_allele_seq1) {
            if (var.ref_allele_seq0.size() > var.ref_allele_seq1.size()) {
                // REF will simply be the longer one.
                ref = var.ref_allele_seq0;

                // Adjust the alt with the short ref allele.
                const std::string &alt_raw = var.alt_allele_seq1;
                alt1 = alt_raw;
                alt1.append(ref.data() + 1, ref.size() - 1);
            } else {
                ref = var.ref_allele_seq1;
                const std::string &alt_raw = var.alt_allele_seq0;
                alt0 = alt_raw;
                alt0.append(ref.data() + 1, ref.size() - 1);
            }
        }
    } else {
        genotype = {var.genotype0[0], var.genotype0[2]};
    }

    return {.is_confident = var.is_confident,
            .is_phased = var.is_phased0,
            .pos = var.pos0,
            .qual = var.is_confident ? 60 : 0,
            .ref = std::string(ref),
            .alts = var.is_multi_allele ? std::vector<std::string>{alt0, alt1}
                                        : std::vector<std::string>{alt0},
            .genotype = genotype};
}

variant_fullinfo_t derive_variant_fullinfo_from_varcall(const ta_t &var,
                                                        const std::string_view refseq_s,
                                                        const uint32_t ref_start,
                                                        const bool allow_N_base) {
    // Note:
    //   - Variant position will be in 0-index.
    variant_fullinfo_t ret;

    std::string ref_s;
    std::string alt_s;
    std::string alt2_s;

    // store ref base
    if (var.pos < ref_start) {  // might happen if pileup used expanded interval
        ret.is_valid = false;
        ret.is_confident = false;
        return ret;
    } else if (var.pos == ref_start) {
        if (!allow_N_base) {
            ret.is_valid = false;
            ret.is_confident = false;
            return ret;
        }
        ref_s = "N";  // store extra 1 base before
        ref_s += refseq_s[var.pos - ref_start];
    } else {
        const int tmppos = var.pos - ref_start;  // 0-index
        if (!allow_N_base) {
            if (refseq_s[tmppos - 1] == 'N' || refseq_s[tmppos] == 'N' ||
                refseq_s[tmppos - 1] == 'n' || refseq_s[tmppos] == 'n') {
                ret.is_valid = false;
                ret.is_confident = false;
                return ret;
            }
        }
        ref_s = refseq_s[tmppos - 1];  // store extra 1 base before
        ref_s += refseq_s[tmppos];
    }

    // store alt base
    int is_del0 = 0;
    int is_ins0 = 0;
    int is_del1 = 0;
    int is_ins1 = 0;
    if (var.type == TA_TYPE_HOM) {
        if (var.alleles[0][0] != SENTINEL_REF_ALLELE_INT) {
            alt_s = nt4seq2seq(var.alleles[0]);
            is_del0 = var.alleles[0].back() == VAR_OP_D;
            is_ins0 = var.alleles[0].back() == VAR_OP_I;
        } else {
            assert(var.alleles.size() > 1);
            alt_s = nt4seq2seq(var.alleles[1]);
            is_del0 = var.alleles[1].back() == VAR_OP_D;
            is_ins0 = var.alleles[1].back() == VAR_OP_I;
        }
    } else {
        if (var.alleles[0][0] != SENTINEL_REF_ALLELE_INT &&
            var.alleles[1][0] != SENTINEL_REF_ALLELE_INT) {  // multi-allele, need to write both
            alt_s = nt4seq2seq(var.alleles[0]);
            alt2_s = nt4seq2seq(var.alleles[1]);
            is_del0 = var.alleles[0].back() == VAR_OP_D;
            is_ins0 = var.alleles[0].back() == VAR_OP_I;
            if (var.alleles.size() > 1) {
                is_del1 = var.alleles[1].back() == VAR_OP_D;
                is_ins1 = var.alleles[1].back() == VAR_OP_I;
            }
        } else {  // (regular case)
            if (var.alleles[0][0] != SENTINEL_REF_ALLELE_INT) {
                alt_s = nt4seq2seq(var.alleles[0]);
                is_del0 = var.alleles[0].back() == VAR_OP_D;
                is_ins0 = var.alleles[0].back() == VAR_OP_I;
            } else {
                alt_s = nt4seq2seq(var.alleles[1]);
                is_del0 = var.alleles[1].back() == VAR_OP_D;
                is_ins0 = var.alleles[1].back() == VAR_OP_I;
            }
        }
    }

    // REF and ALT
    if (var.is_used == TA_STAT_ACCEPTED) {  // "PASS"
        ret.is_confident = true;
        ret.qual0 = 60;
        ret.qual1 = 60;
    } else {  // "unsr"
        ret.is_confident = false;
        ret.qual0 = 0;
        ret.qual1 = 0;
    }

    if (is_del0) {
        ret.pos0 = var.pos - 1;  // 0-index
        ret.ref_allele_seq0 = ref_s[0] + alt_s;
        ret.alt_allele_seq0 = ref_s[0];
    } else if (is_ins0) {
        ret.pos0 = var.pos - 1;  // 0-index
        ret.ref_allele_seq0 = ref_s[0];
        ret.alt_allele_seq0 = ref_s[0] + alt_s;
    } else {
        ret.pos0 = var.pos;  // 0-index
        ret.ref_allele_seq0 = ref_s.c_str() + 1;
        ret.alt_allele_seq0 = alt_s;
    }
    ret.genotype0[0] = var.genotype[0];
    ret.genotype0[1] = var.genotype[1];
    ret.genotype0[2] = var.genotype[2];
    ret.is_phased0 = var.genotype[1] == '|';
    ret.pos1 = 0;
    ret.qual1 = 0;

    ret.is_multi_allele = (var.type == TA_TYPE_HETMULTI);
    if (ret.is_multi_allele) {
        if (is_del1) {
            ret.pos1 = var.pos - 1;  // 0-index
            ret.ref_allele_seq1 = ref_s[0] + alt2_s;
            ret.alt_allele_seq1 = ref_s[0];
        } else if (is_ins1) {
            ret.pos1 = var.pos - 1;  // 0-index
            ret.ref_allele_seq1 = ref_s[0];
            ret.alt_allele_seq1 = ref_s[0] + alt2_s;
        } else {
            ret.pos1 = var.pos;  // 0-index
            ret.ref_allele_seq1 = ref_s.c_str() + 1;
            ret.alt_allele_seq1 = alt2_s;
        }
        ret.genotype1[0] = var.genotype[2];
        ret.genotype1[1] = var.genotype[1];
        ret.genotype1[2] = var.genotype[0];
        ret.is_phased1 = var.genotype[1] == '|';
    }

    return ret;
}

void fix_variant_fullinfo_genotype_snp_in_del(std::vector<variant_fullinfo_t> &vars) {
    // Note (variant representation):
    //    If at a position, one hap has a substitution while the other hap
    //    has a deletion that started prior to this position & extends to
    //    cover it, it may be preferrable to mark the substitution's genotype
    //    as hom. For an example,  see HG002 chr6:1128114 .
    //    `ta_t` saves this case as het,
    //    because the other hap doesn't have the ref base and thus is different.
    //    Let's convert it to hom here.
    for (int64_t i = 1; i < std::ssize(vars); i++) {
        const auto &prev_var = vars[i - 1];
        auto &var = vars[i];
        bool should_set_to_hom = false;

        if (var.genotype0[0] == var.genotype0[2]) {
            continue;
        }  // already hom
        if (!var.is_valid || !prev_var.is_valid) {
            continue;
        }
        if (!prev_var.is_confident) {
            continue;
        }
        if (var.is_multi_allele) {
            continue;
        }
        if (var.ref_allele_seq0.size() != var.alt_allele_seq0.size()) {
            continue;
        }

        if (prev_var.ref_allele_seq0.size() > prev_var.alt_allele_seq0.size()) {  // is del
            const uint32_t del_len = static_cast<uint32_t>(prev_var.ref_allele_seq0.size());
            if (prev_var.pos0 + del_len >= var.pos0) {
                should_set_to_hom = true;
            }
        }
        if (prev_var.is_multi_allele) {
            if (prev_var.ref_allele_seq1.size() > prev_var.alt_allele_seq1.size()) {  // is del
                const uint32_t del_len = static_cast<uint32_t>(prev_var.ref_allele_seq1.size());
                if (prev_var.pos1 + del_len >= var.pos1) {
                    should_set_to_hom = true;
                }
            }
        }

        if (should_set_to_hom) {
            if constexpr (DEBUG_LOCAL_HAPLOTAGGING) {
                LOG_DEBUG(
                        "[{}] substitution het candidate at pos {} shadowed by del ({:d} {}->{}), "
                        "setting to hom.",
                        __func__, static_cast<int>(var.pos0) + 1,
                        static_cast<int>(prev_var.pos0) + 1, prev_var.ref_allele_seq0,
                        prev_var.alt_allele_seq0);
            }
            var.genotype0[0] = '1';
            var.genotype0[1] = '/';  // also marking as unphased
            var.genotype0[2] = '1';
        }
    }
}
}  // namespace

std::string create_region_string(const std::string_view ref_name,
                                 const uint32_t start,
                                 const uint32_t end) {
    std::string ret(ref_name);
    ret.append(":");
    ret.append(std::to_string(start + 1));
    ret.append("-");
    ret.append(std::to_string(end));
    return ret;
}

chunk_t variant_pileup_ht(BamFileView &hf,
                          const variants_t &ht_refvars,
                          const faidx_t *fai,
                          const str2int_t *qname2hp,  // 0-index
                          const std::string_view refname,
                          const uint32_t itvl_start,
                          const uint32_t itvl_end,
                          const pileup_pars_t &pp) {
    // A simpler pileup that allows hom variants, non-SNPs and multi-alleles.
    // This is for variant calling.

    const bool enable_downsample = true;
    const int downsample_window = 10000;
    const int downsample_readcap = 150;  // 10k window 30x has ~50 reads

    if constexpr (DEBUG_LOCAL_HAPLOTAGGING) {
        LOG_DEBUG("[kdys::{}] pileup at {}:{}-{} (1-index, close-open)", __func__, refname.data(),
                  itvl_start + 1, itvl_end);
    }
    vc_variants1_t ht;

    chunk_t ck = {
            .is_valid = true,
            .reads = {},
            .varcalls = {},
            .qnames = {},
            .qname2ID = {},
            .abs_start = static_cast<uint32_t>(itvl_start),
            .abs_end = static_cast<uint32_t>(itvl_end),
            .refname = std::string(refname),
            .vg = {},
    };
    uint32_t abs_start = ck.abs_start;
    uint32_t abs_end = ck.abs_end;

    const std::string itvl = create_region_string(refname, itvl_start, itvl_end);
    HtsItrPtr bamitr = HtsItrPtr(sam_itr_querys(hf.idx, hf.hdr, itvl.c_str()), HtsItrDestructor());
    BamPtr aln = BamPtr(bam_init1(), BamDestructor());

    if (!pp.disable_region_expansion) {
        interval_t new_itvl = expand_query_interval(hf, refname, itvl_start, itvl_end);
        abs_start = new_itvl.start;
        abs_end = new_itvl.end;
        const std::string itvl2 = create_region_string(refname, abs_start, abs_end);

        bamitr = HtsItrPtr(sam_itr_querys(hf.idx, hf.hdr, itvl2.c_str()), HtsItrDestructor());
        if constexpr (DEBUG_LOCAL_HAPLOTAGGING) {
            LOG_DEBUG("[kdys::{}] expanded: {} (disable_interval_expansion={}", __func__, itvl2,
                      pp.disable_region_expansion);
        }
    }

    BlockedBloomFilter bf(4, 25);
    if (pp.use_bloomfilter) {
        bf.enable();  // allocates
    }

    // downsample helper
    int downsample_filtered = 0;
    std::vector<int> downsample_counter;
    const int n_counter = (abs_end - abs_start) / downsample_window + 1;
    if (enable_downsample) {
        downsample_counter.resize(n_counter, 0);
    }
    uint32_t n_reads = 0;
    bool pileup_failed = false;
    while (sam_itr_next(hf.fp, bamitr.get(), aln.get()) >= 0) {
        const char *qn = bam_get_qname(aln.get());

        if (n_reads > MAX_READS) {
            spdlog::error(
                    "[kdys::{}] too many reads (after downsampling), giving up the whole chunk.",
                    __func__);
            pileup_failed = true;
            break;
        }
        if constexpr (DEBUG_LOCAL_HAPLOTAGGING) {
            if (n_reads % 1000 == 0) {
                LOG_DEBUG(
                        "[kdys::{}] piled {} reads (read start pos is {}) ht size {}, seen size {}",
                        __func__, n_reads, aln.get()->core.pos, ht.size(), ck.reads.size());
            }
        }

        const int flag = aln.get()->core.flag;
        const int mapq = (int)aln->core.qual;
        float de = 0;
        uint8_t *tmp = bam_aux_get(aln.get(), "de");
        if (tmp) {
            de = static_cast<float>(bam_aux2f(tmp));
        }

        const bool md_is_ok = sancheck_MD_tag_exists_and_is_valid(aln.get());
        if (!md_is_ok) {
            continue;
        }
        if (aln.get()->core.n_cigar == 0) {
            continue;
        }
        if ((flag & 4) || (flag & 256) || (flag & 2048)) {
            continue;
        }
        if (mapq < pp.min_mapq) {
            continue;
        }
        if (de > pp.max_gapcompressed_seqdiv) {
            continue;
        }
        uint8_t hp = HAPTAG_UNPHASED;
        if (qname2hp) {
            const auto it = qname2hp->find(qn);
            if (it == qname2hp->cend()) {
                hp = HAPTAG_UNPHASED;
            } else {
                hp = static_cast<uint8_t>(it->second);
            }
        }
        const uint32_t r_start_pos = static_cast<uint32_t>(aln.get()->core.pos);
        const uint32_t r_end_pos = static_cast<uint32_t>(bam_endpos(aln.get()));

        if (enable_downsample) {
            const uint32_t effective_r_start =
                    (r_start_pos > abs_start ? r_start_pos - abs_start : 0);
            const uint32_t effective_r_end = (r_end_pos > abs_end ? abs_end : r_end_pos) -
                                             (r_start_pos > abs_start ? r_start_pos : abs_start);
            const uint32_t iw_s = effective_r_start /
                                  downsample_window;  // itvl was 1-index while bam itr is 0 index..
            if (downsample_counter[iw_s] < downsample_readcap) {
                // update the counter of downsampling
                uint32_t iw_n = effective_r_end / downsample_window;
                for (uint32_t tmpi = iw_s; tmpi < iw_s + iw_n; tmpi++) {
                    downsample_counter[tmpi] = downsample_counter[tmpi] < INT_MAX
                                                       ? downsample_counter[tmpi] + 1
                                                       : downsample_counter[tmpi];
                }
            } else {
                downsample_filtered++;
                continue;  // go parse the next read
            }
        }

        n_reads++;
        // collect variants of the read
        read_t r{
                .start_pos = static_cast<uint32_t>(aln.get()->core.pos),
                .end_pos = static_cast<uint32_t>(bam_endpos(aln.get())),
                .ID = n_reads,
                .vars = {},
                .hp = hp,
                .votes_diploid = {0, 0},
                .strand = !!(flag & 16),
                .de = de,
                .left_clip_len = 0,
                .right_clip_len = 0,
        };
        // (if we have a list of trusted variants, the read's variants should
        // be stored in a temporary buffer first and be filtered)
        // TODO could be done in parse_variant_for_one_read instead.
        std::vector<qa_t> tmp_read_vars_buffer{};
        std::vector<qa_t> &read_vars_buffer = ht_refvars.empty() ? r.vars : tmp_read_vars_buffer;
        const bool parse_ok = parse_variants_for_one_read(
                aln.get(), read_vars_buffer, pp.min_base_quality, &r.left_clip_len,
                &r.right_clip_len, pp.retain_SNP_only, nullptr);
        if (parse_ok && !read_vars_buffer.empty() && r.left_clip_len < pp.max_clipping &&
            r.right_clip_len < pp.max_clipping) {
            if (!ht_refvars.empty()) {
                filter_lift_qa_v_given_conf_list(read_vars_buffer, r.vars, ht_refvars);
            }
            for (uint32_t i = 0; i < r.vars.size(); i++) {
                if (r.vars[i].pos < abs_start) {
                    continue;
                }
                if (r.vars[i].pos >= abs_end) {
                    break;
                }
                const int is_not_seen_before =
                        pp.use_bloomfilter ? bf.insert(r.vars[i].pos) : false;
                if (is_not_seen_before) {
                    continue;
                }

                const qa_t &var = r.vars[i];

                // check if allele exists
                int allele_found = 0;
                if (ht.find(var.pos) != ht.end()) {
                    for (uint32_t j = 0; j < ht[var.pos].alleles.size(); j++) {
                        vc_allele_t &allele = ht[var.pos].alleles[j];
                        if (allele.allele == var.allele) {
                            // update overall coverage
                            if (hp == 0) {
                                allele.cov.cov_hap0 += 1;
                            } else if (hp == 1) {
                                allele.cov.cov_hap1 += 1;
                            } else {
                                allele.cov.cov_unphased += 1;
                            }
                            // update per-strand coverage
                            if (r.strand == 0) {
                                allele.cov.cov_fwd += 1;
                            } else {
                                allele.cov.cov_bwd += 1;
                            }
                            allele_found = 1;
                            break;
                        }
                    }
                }

                if (!allele_found) {
                    ht[var.pos].is_accepted = FLAG_VARSTAT_UNKNOWN;
                    ht[var.pos].alleles.push_back({{.cov_hap0 = 0,
                                                    .cov_hap1 = 0,
                                                    .cov_unphased = 0,
                                                    .cov_tot_all = 0,
                                                    .cov_tot_phased = 0,
                                                    .cov_fwd = 0,
                                                    .cov_bwd = 0},
                                                   var.allele});
                    vc_allele_t &allele = ht[var.pos].alleles.back();
                    if (hp == 0) {
                        allele.cov.cov_hap0 += 1;
                    } else if (hp == 1) {
                        allele.cov.cov_hap1 += 1;
                    } else {
                        allele.cov.cov_unphased += 1;
                    }
                    // update per-strand coverage
                    if (r.strand == 0) {
                        allele.cov.cov_fwd += 1;
                    } else {
                        allele.cov.cov_bwd += 1;
                    }
                }
            }

            ck.reads.emplace_back(std::move(r));
            ck.qnames.emplace_back(qn);
        }
    }  // iterate through read alignments

    if constexpr (DEBUG_LOCAL_HAPLOTAGGING) {
        spdlog::debug("[{}] ht size is {}, ck size {} (downsample={}, filtered={}), n_reads={}",
                      __func__, ht.size(), ck.reads.size(), enable_downsample, downsample_filtered,
                      n_reads);
    }

    if (pileup_failed) {
        spdlog::error(
                "[kdys::{}] query {}:{}-{} (1-index [) ]) pileup's initial collection failed.",
                __func__, refname.data(), itvl_start + 1, itvl_end);
        return {};
    }

    if (ck.qnames.size() != ck.reads.size()) {
        spdlog::error(
                "[{}] ck qnames and reads buffers have different lenghts ({} and {}), "
                "impossible, check code",
                __func__, static_cast<int>(ck.qnames.size()), static_cast<int>(ck.reads.size()));
        return {};
    }

    // Unphased pileup needs low complexity masking, otherwise
    // phasing could be wrong. We will miss some real het variants
    // this way, but it is safer, as the phased pileup varcall has
    // little recourse against confidently wrong phasing.
    // (TODO: add phasing validation?)
    std::vector<uint64_t> hplowcmp_mask;
    std::string refseq_s;
    int refseq_l = 0;
    if (fai) {
        if (!pp.disable_low_complexity_masking) {
            hplowcmp_mask = get_lowcmp_mask(fai, refname, abs_start, abs_end);
        }
        const std::string span_s = create_region_string(refname, abs_start, abs_end);
        refseq_s = kadayashi::hts_utils::fetch_seq(fai, span_s.c_str());
        refseq_l = static_cast<int>(refseq_s.size());
    }

    // initial parsing: filter by ALT variant's coverage
    std::vector<uint32_t> candidate_poss;
    for (auto &[pos, q] : ht) {
        int alt_cov = 0;
        int n_passed = 0;
        for (uint32_t i = 0; i < q.alleles.size(); i++) {
            cov_t &cov = q.alleles[i].cov;
            alt_cov += cov.cov_hap0 + cov.cov_hap1 + cov.cov_unphased;
            if (cov.cov_fwd >= 2 && cov.cov_bwd >= 2) {
                n_passed++;
            }
        }
        bool passed_lowcmp = qname2hp ? true : !is_lowcmp_masked(hplowcmp_mask, pos);
        if ((!qname2hp && n_passed >= 1 && passed_lowcmp) ||
            (qname2hp &&
             alt_cov >= 3)) {  // we will try to collect REF; push the ref allele placeholders now
            q.is_accepted = FLAG_VARSTAT_MAYBE;
            q.alleles.push_back({{.cov_hap0 = 0,
                                  .cov_hap1 = 0,
                                  .cov_unphased = 0,
                                  .cov_tot_all = 0,
                                  .cov_tot_phased = 0,
                                  .cov_fwd = 0,
                                  .cov_bwd = 0},
                                 {SENTINEL_REF_ALLELE_INT, VAR_OP_D}});
            q.alleles.push_back({{.cov_hap0 = 0,
                                  .cov_hap1 = 0,
                                  .cov_unphased = 0,
                                  .cov_tot_all = 0,
                                  .cov_tot_phased = 0,
                                  .cov_fwd = 0,
                                  .cov_bwd = 0},
                                 {SENTINEL_REF_ALLELE_INT, VAR_OP_X}});
            candidate_poss.push_back(pos);
            if constexpr (DEBUG_LOCAL_HAPLOTAGGING) {
                LOG_DEBUG("[kdys::{}] pos {} will collect ref ({} pileup)", __func__, pos,
                          qname2hp ? "phased" : "unphased");
            }
        } else {
            q.is_accepted = FLAG_VARSTAT_REJECTED;
        }
    }
    if (candidate_poss.empty()) {
        ck.reads.clear();
        ck.is_valid = false;
        return ck;
    }

    std::sort(candidate_poss.begin(), candidate_poss.end());
    for (uint32_t i_read = 0; i_read < ck.reads.size(); i_read++) {
        std::stable_sort(ck.reads[i_read].vars.begin(), ck.reads[i_read].vars.end());
    }

    // 2nd parsing: collect REF allele for candidates
    // TODO: should discount any deletions that overlap with SNP with ALT,
    //       otherwise here we risk calling ALT/REF when it is actually ALT/. genotype.
    for (uint32_t i_read = 0; i_read < ck.reads.size(); i_read++) {
        read_t &read = ck.reads[i_read];
        const int hp = (int)read.hp;
        const uint32_t aln_start = read.start_pos;
        const uint32_t aln_end = read.end_pos;

        // get range on known alts
        uint32_t i_lower = static_cast<uint32_t>(std::distance(
                candidate_poss.begin(),
                std::lower_bound(candidate_poss.begin(), candidate_poss.end(), aln_start)));
        if (i_lower >= candidate_poss.size()) {
            continue;
        }
        if (candidate_poss[i_lower] !=
            aln_start) {  // if not equal, get less-than rather than no-less-than
            i_lower = i_lower == 0 ? 0 : i_lower - 1;
        }
        uint32_t i_higher = static_cast<uint32_t>(std::distance(
                candidate_poss.begin(),
                std::upper_bound(candidate_poss.begin(), candidate_poss.end(), aln_end)));
        if (i_higher > candidate_poss.size()) {
            i_higher = static_cast<uint32_t>(candidate_poss.size());
        }

        if constexpr (DEBUG_LOCAL_HAPLOTAGGING) {
            LOG_DEBUG("[kdys::{}] qn {} candidate alts bewtween: {} - {} (aln_end is {})", __func__,
                      ck.qnames[i_read], candidate_poss[i_lower], candidate_poss[i_higher],
                      aln_end);
        }

        // scan
        for (uint32_t i = i_lower, j = 0; i < i_higher && j < read.vars.size(); /****/) {
            while (candidate_poss[i] < aln_start) {
                i++;
                if (i >= i_higher) {
                    break;
                }
            }
            if (candidate_poss[i] >= aln_end || i >= i_higher) {
                break;
            }
            const int read_prev_var_was_del = (j > 0 && read.vars[j - 1].allele.back() == VAR_OP_D);
            const int last_del_size =
                    read_prev_var_was_del ? (int)std::ssize(read.vars[j - 1].allele) - 1 : 0;
            while (candidate_poss[i] < read.vars[j].pos) {
                vc_allele_t *aa = nullptr;
                int is_del_case = 0;
                if (ht[candidate_poss[i]].alleles.size() > 0 &&
                    (!read_prev_var_was_del ||
                     (read_prev_var_was_del &&
                      read.vars[j - 1].pos + last_del_size <
                              candidate_poss[i]))) {  // log a ref-substitute
                    assert(ht[candidate_poss[i]].alleles.back().allele[0] ==
                           SENTINEL_REF_ALLELE_INT);
                    assert(ht[candidate_poss[i]].alleles.back().allele[1] == VAR_OP_X);
                    aa = &ht[candidate_poss[i]].alleles.back();
                } else {  // log a ref-del instead
                    aa = &ht[candidate_poss[i]].alleles[ht[candidate_poss[i]].alleles.size() - 2];
                    is_del_case = 1;
                }

                if (hp == 0) {
                    aa->cov.cov_hap0++;
                } else if (hp == 1) {
                    aa->cov.cov_hap1++;
                } else {
                    aa->cov.cov_unphased++;
                }
                if (read.strand == 0) {
                    aa->cov.cov_fwd++;
                } else {
                    aa->cov.cov_bwd++;
                }
                add_allele_qa_v_nt4seq(read.vars, candidate_poss[i],
                                       std::vector<uint8_t>{SENTINEL_REF_ALLELE_INT},
                                       SENTINEL_REF_ALLELE_L, is_del_case ? VAR_OP_D : VAR_OP_X);
                i++;
                if (!(i < i_higher && j < read.vars.size())) {
                    break;
                }
            }
            if (!(i < i_higher && j < read.vars.size())) {
                break;
            }
            if (candidate_poss[i] == read.vars[j].pos) {
                i++;
                j++;
            } else {
                j++;
            }
        }
        std::stable_sort(read.vars.begin(), read.vars.end());
    }  // iter through reads

    // finish filling in coverages
    for (auto &[pos, q] : ht) {
        variant_fill_cov_tot(q);
    }

    // 3rd parsing: decide candidates
    // There's three cases for diploid: hom ALT/ALT, het REF/ALT, and HET1/HET2
    for (auto &[pos, q] : ht) {
        // (prep: sort by pileup coverage in reverse order)
        std::stable_sort(q.alleles.begin(), q.alleles.end(),
                         [](const vc_allele_t &a, const vc_allele_t &b) {
                             return a.cov.cov_hap0 + a.cov.cov_hap1 + a.cov.cov_unphased >=
                                    b.cov.cov_hap0 + b.cov.cov_hap1 + b.cov.cov_unphased;
                         });

        const bool is_done =
                classify_variant_prefilter(q, pos, refseq_s.c_str(), refseq_l, abs_start, qname2hp);
        if (is_done) {
            continue;
        }

        q.is_accepted = FLAG_VARSTAT_REJECTED;  // default to reject
        q.type = FLAG_VAR_NA;
        if (qname2hp) {  // phased varcall
            assert(fai);
            const var_classify_t result =
                    classify_variant_phased(q, pos, refseq_s.c_str(), refseq_l, abs_start,
                                            pp.min_strand_cov, pp.min_strand_cov_frac);
            q.is_accepted = result.is_accepted;
            q.type = result.type;
        } else {  // unphased varcall: either accept or reject, no middle ground
            const int c0 = q.alleles[0].cov.cov_tot_all;
            const int c1 = q.alleles[1].cov.cov_tot_all;
            const float c0c1 = (c0 == 0 && c1 == 0) ? 1.0f : static_cast<float>(c0 + c1);
            const int threshold = static_cast<int>(
                    std::max(static_cast<int>((float)(c0 + c1) * pp.min_varcall_fraction),
                             pp.min_varcall_coverage));
            const float ratio = static_cast<float>(std::min(c0, c1)) / c0c1;

            if (c0 >= threshold && c1 >= threshold && ratio >= pp.min_varcall_fraction) {
                q.is_accepted = FLAG_VARSTAT_ACCEPTED;
                if (q.alleles[0].allele[0] == SENTINEL_REF_ALLELE_INT ||
                    q.alleles[1].allele[0] == SENTINEL_REF_ALLELE_INT) {
                    q.type = FLAG_VAR_HET;
                } else {
                    q.type = FLAG_VAR_MULTHET;
                }
            } else {
                if (c0 >= 3 && c1 < 3 && q.alleles[0].allele[0] != SENTINEL_REF_ALLELE_INT) {
                    q.is_accepted = FLAG_VARSTAT_ACCEPTED;
                    q.type = FLAG_VAR_HOM;
                } else {
                    q.is_accepted = FLAG_VARSTAT_REJECTED;
                }
            }
            if constexpr (DEBUG_LOCAL_HAPLOTAGGING) {
                LOG_DEBUG(
                        "[{}] unphased pileup pos {} : c0={} ({:c}) c1={} ({:c}) ratio={:.2f} -> "
                        "type={} "
                        "is_accepted={}",
                        __func__, pos, c0, "ACGT_R"[q.alleles[0].allele[0]], c1,
                        "ACGT_R"[q.alleles[1].allele[0]], ratio, q.type, q.is_accepted);
            }
        }
    }  // iter through variant candidate positions

    // store: variants positions and alleles
    std::vector<valid_pos_t> valid_poss;
    std::unordered_map<uint32_t, uint32_t> valid_poss_set;
    std::vector<uint32_t> tmp_sorted_poss;
    for (const auto &[pos, q] : ht) {
        tmp_sorted_poss.push_back(pos);
    }
    std::sort(tmp_sorted_poss.begin(), tmp_sorted_poss.end());
    for (size_t i_pos = 0; i_pos < tmp_sorted_poss.size(); i_pos++) {
        const uint32_t pos = tmp_sorted_poss[i_pos];
        const auto &q = ht[pos];
        const uint8_t var_stat = q.is_accepted;
        if ((var_stat & (FLAG_VARSTAT_ACCEPTED | FLAG_VARSTAT_UNSURE))) {
            if (pp.retain_het_only && q.type != FLAG_VAR_HET) {
                continue;
            }
            if (valid_poss.size() > 0 && pos == valid_poss.back().pos) {
                // TODO: handle multiple variants at one spot here after the variant
                // filtering impl above becomes able to handle them.
                continue;
            }
            valid_poss.push_back({.pos = pos, .stat = var_stat});
        }
    }
    for (size_t i = 0; i < valid_poss.size(); i++) {
        valid_poss_set[valid_poss[i].pos] = static_cast<uint32_t>(i);
    }
    for (const auto pos_stat : valid_poss) {
        const uint32_t pos = pos_stat.pos;
        const uint8_t var_stat = pos_stat.stat;
        const uint8_t var_type = ht[pos].type;
        if (var_stat != FLAG_VARSTAT_ACCEPTED && var_stat != FLAG_VARSTAT_UNSURE) {
            continue;
        }

        ck.varcalls.push_back(ta_t{});
        ck.varcalls.back().pos = pos;

        if (var_stat == FLAG_VARSTAT_ACCEPTED) {
            ck.varcalls.back().is_used = TA_STAT_ACCEPTED;
        } else {
            ck.varcalls.back().is_used = TA_STAT_UNSURE;
        }

        if (var_stat == FLAG_VARSTAT_UNSURE) {  // save all alleles
            for (auto &_ : ht[pos].alleles) {
                ck.varcalls.back().alleles.push_back(_.allele);
            }
            if (var_type == FLAG_VAR_MULTHET) {
                ck.varcalls.back().type = TA_TYPE_HETMULTI;
            } else if (var_type == FLAG_VAR_HET) {
                ck.varcalls.back().type = TA_TYPE_HET;
            } else if (var_type == FLAG_VAR_HOM) {
                ck.varcalls.back().type = TA_TYPE_HOM;
            } else {
                ck.varcalls.back().type = TA_TYPE_UNKNOWN;
            }
        } else {
            if (var_type != FLAG_VAR_MULTHET) {
                if (var_type == FLAG_VAR_HOM) {
                    assert(ht[pos].alleles[0].allele[0] != SENTINEL_REF_ALLELE_INT);
                    ck.varcalls.back().alleles.push_back(ht[pos].alleles[0].allele);
                    ck.varcalls.back().type = TA_TYPE_HOM;
                } else {
                    assert(var_type == FLAG_VAR_HET);
                    ck.varcalls.back().alleles.push_back(ht[pos].alleles[0].allele);
                    ck.varcalls.back().alleles.push_back(ht[pos].alleles[1].allele);
                    ck.varcalls.back().type = TA_TYPE_HET;
                }
            } else {
                assert(ht[pos].alleles[0].allele[0] != SENTINEL_REF_ALLELE_INT);
                assert(ht[pos].alleles[1].allele[0] != SENTINEL_REF_ALLELE_INT);
                ck.varcalls.back().alleles.push_back(ht[pos].alleles[0].allele);
                ck.varcalls.back().alleles.push_back(ht[pos].alleles[1].allele);
                ck.varcalls.back().type = TA_TYPE_HETMULTI;
            }
        }
    }

    // store: variants on reads and reads of a given variants
    std::vector<qa_t> new_;
    for (uint32_t i_read = 0; i_read < ck.reads.size(); i_read++) {
        //auto &r = reads_seen_positions[i_read];
        read_t &read = ck.reads[i_read];
        new_.clear();
        for (uint32_t i = 0; i < read.vars.size(); i++) {
            const uint32_t pos = read.vars[i].pos;
            const std::vector<uint8_t> &allele = read.vars[i].allele;
            if (valid_poss_set.find(pos) != valid_poss_set.end()) {
                ta_t &ta = ck.varcalls[valid_poss_set[pos]];

                // find the allele in buffer
                int allele_idx = -1;
                if (ta.alleles[0] == allele) {
                    allele_idx = 0;
                } else if (ta.alleles.size() > 1 && ta.alleles[1] == allele) {
                    allele_idx = 1;
                }
                if (allele_idx >= 0) {
                    if (static_cast<uint32_t>(allele_idx) + 1 > ta.allele2readIDs.size()) {
                        ta.allele2readIDs.resize(allele_idx + 1);
                    }

                    // log the read ID
                    ta.allele2readIDs[allele_idx].push_back(i_read);

                    // log the read var
                    new_.push_back(qa_t{.pos = pos,
                                        .allele = allele,
                                        .var_idx = valid_poss_set[pos],
                                        .allele_idx = static_cast<uint32_t>(allele_idx),
                                        .is_used = ta.is_used,
                                        .hp = HAPTAG_UNPHASED});
                }
            }
        }

        // let read's var buffer to only have called variants
        read.vars.clear();
        read.vars.insert(read.vars.end(), new_.begin(), new_.end());
    }

    return ck;
}

std::unordered_map<std::string, int> kadayashi_local_haptagging_gen_ht(chunk_t &ck) {
    std::unordered_map<std::string, int> qname2hp;
    for (size_t i = 0; i < ck.reads.size(); i++) {
        const int haptag = ck.reads[i].hp;
        qname2hp[ck.qnames[i]] = haptag;
    }
    return qname2hp;
}

bool operator==(const variant_fullinfo_t &a, const variant_fullinfo_t &b) {
    return std::tie(a.is_confident, a.is_multi_allele, a.pos0, a.qual0, a.ref_allele_seq0,
                    a.alt_allele_seq0, a.is_phased0, a.genotype0[0], a.genotype0[1], a.genotype0[2],
                    a.pos1, a.qual1, a.ref_allele_seq1, a.alt_allele_seq1, a.is_phased1,
                    a.genotype1[0], a.genotype1[1], a.genotype1[2]) ==
           std::tie(b.is_confident, b.is_multi_allele, b.pos0, b.qual0, b.ref_allele_seq0,
                    b.alt_allele_seq0, b.is_phased0, b.genotype0[0], b.genotype0[1], b.genotype0[2],
                    b.pos1, b.qual1, b.ref_allele_seq1, b.alt_allele_seq1, b.is_phased1,
                    b.genotype1[0], b.genotype1[1], b.genotype1[2]);
};

bool operator==(const variant_dorado_style_t &a, const variant_dorado_style_t &b) {
    constexpr bool DEBUG_PRINT = false;
    if constexpr (DEBUG_PRINT) {
        LOG_TRACE(
                "[variant_dorado_style_t=] conf={} phased={} pos={} qual={} ref={} alt={},{} "
                "geno={:c},{:c}",
                a.is_confident ? "true" : "false", a.is_phased ? "true" : "false", a.pos, a.qual,
                a.ref, a.alts[0], a.alts.size() > 1 ? a.alts[1] : "", a.genotype.first,
                a.genotype.second);
    }
    return std::tie(a.is_confident, a.is_phased, a.pos, a.qual, a.ref, a.alts, a.genotype) ==
           std::tie(b.is_confident, b.is_phased, b.pos, b.qual, b.ref, b.alts, b.genotype);
};

phase_return_t kadayashi_local_haptagging_dvr_single_region(samFile *fp_bam,
                                                            hts_idx_t *fp_bai,
                                                            sam_hdr_t *fp_header,
                                                            const faidx_t *fai,
                                                            const std::string_view ref_name,
                                                            const uint32_t ref_start,
                                                            const uint32_t ref_end,
                                                            const pileup_pars_t &pp) {
    // Return:
    //    Unless hashtable's initialization or input opening failed,
    //     this function always return a hashtable, which might be empty
    //     if no read can be tagged or there was any error
    //     when tagging reads.
    //    Haptags in the hashtable is 0-index.

    phase_return_t ret;

    BamFileView hf{.fp = fp_bam, .idx = fp_bai, .hdr = fp_header};

    chunk_t ck = variant_pileup_ht(hf, {}, fai, nullptr, ref_name, ref_start, ref_end, pp);

    const bool variant_graph_ok = variant_graph_gen(ck);
    if (variant_graph_ok) {
        variant_graph_propogate(ck);
        variant_graph_haptag_reads(ck);
        ret.qname2hp = kadayashi_local_haptagging_gen_ht(ck);
        if (ck.vg.has_breakpoints) {
            for (uint32_t i_var = 0; i_var < ck.vg.n_vars; i_var++) {
                if (ck.vg.next_link_is_broken[i_var]) {
                    const uint32_t var_pos = ck.varcalls[i_var].pos;
                    ret.phasing_breakpoints[var_pos] = 1;
                }
            }
        }
    }
    ret.ck = std::move(ck);

    return ret;
}

phase_return_t kadayashi_local_haptagging_simple_single_region(samFile *fp_bam,
                                                               hts_idx_t *fp_bai,
                                                               sam_hdr_t *fp_header,
                                                               const faidx_t *fai,
                                                               const std::string_view ref_name,
                                                               const uint32_t ref_start,
                                                               const uint32_t ref_end,
                                                               const pileup_pars_t &pp) {
    phase_return_t ret;

    BamFileView hf{.fp = fp_bam, .idx = fp_bai, .hdr = fp_header};

    chunk_t ck = variant_pileup_ht(hf, {}, fai, nullptr, ref_name, ref_start, ref_end, pp);

    const bool variant_graph_ok = variant_graph_gen(ck);
    if (variant_graph_ok) {
        const int n_iter = std::max(10, static_cast<int>(ref_end - ref_start) / 10000);
        variant_graph_do_simple_haptag(ck, n_iter);
        for (size_t i = 0; i < ck.reads.size(); i++) {
            const int haptag = ck.reads[i].hp;
            ret.qname2hp[ck.qnames[i]] = haptag;
        }

        uint32_t phased_until = 0;
        for (uint32_t readID = 0; readID < ck.reads.size(); readID++) {
            const auto &read = ck.reads[readID];
            if (read.vars.size() == 0) {
                continue;
            }
            const uint8_t hp = read.hp;

            bool broke = false;
            if (hp != HAPTAG_UNPHASED) {
                if (read.vars[0].pos >= phased_until) {
                    broke = true;
                } else {
                    broke = false;
                }
                phased_until = std::max<uint32_t>(phased_until, read.vars.back().pos);
            } else {
                continue;
            }

            if (broke) {
                // take the leftmost variant's position of the
                // first unphased read as the phasing breakpoint.
                const uint32_t pos = read.vars[0].pos;
                ret.phasing_breakpoints[pos] = 1;
            }
        }
    }
    ret.ck = std::move(ck);

    return ret;
}

std::unordered_map<std::string, int> kadayashi_dvr_single_region_wrapper(
        samFile *fp_bam,
        hts_idx_t *fp_bai,
        sam_hdr_t *fp_header,
        const faidx_t *fai,
        const std::string_view ref_name,
        const uint32_t ref_start,
        const uint32_t ref_end,
        const bool disable_interval_expansion,
        const int min_base_quality,
        const int min_varcall_coverage,
        const float min_varcall_fraction,
        const int max_clipping,
        const int min_strand_cov,
        const float min_strand_cov_frac,
        const float max_gapcompressed_seqdiv) {
    auto result = kadayashi_dvr_single_region_wrapper1(
            fp_bam, fp_bai, fp_header, fai, ref_name, ref_start, ref_end,
            disable_interval_expansion, min_base_quality, min_varcall_coverage,
            min_varcall_fraction, max_clipping, min_strand_cov, min_strand_cov_frac,
            max_gapcompressed_seqdiv);

    return std::move(result.qname2hp);
}

std::unordered_map<std::string, int> kadayashi_simple_single_region_wrapper(
        samFile *fp_bam,
        hts_idx_t *fp_bai,
        sam_hdr_t *fp_header,
        const faidx_t *fai,
        const std::string_view ref_name,
        const uint32_t ref_start,
        const uint32_t ref_end,
        const bool disable_interval_expansion,
        const int min_base_quality,
        const int min_varcall_coverage,
        const float min_varcall_fraction,
        const int max_clipping,
        const int min_strand_cov,
        const float min_strand_cov_frac,
        const float max_gapcompressed_seqdiv) {
    auto result = kadayashi_simple_single_region_wrapper1(
            fp_bam, fp_bai, fp_header, fai, ref_name, ref_start, ref_end,
            disable_interval_expansion, min_base_quality, min_varcall_coverage,
            min_varcall_fraction, max_clipping, min_strand_cov, min_strand_cov_frac,
            max_gapcompressed_seqdiv);

    return std::move(result.qname2hp);
}

ck_and_varcall_result_t kadayashi_phase_and_varcall(samFile *fp_bam,
                                                    hts_idx_t *fp_bai,
                                                    sam_hdr_t *fp_header,
                                                    const faidx_t *fai,
                                                    const std::string_view ref_name,
                                                    const uint32_t ref_start,
                                                    const uint32_t ref_end,
                                                    const bool disable_interval_expansion,
                                                    const int min_base_quality,
                                                    const int min_varcall_coverage,
                                                    const float min_varcall_fraction,
                                                    const int max_clipping,
                                                    const int min_strand_cov,
                                                    const float min_strand_cov_frac,
                                                    const float max_gapcompressed_seqdiv,
                                                    const bool use_dvr_for_phasing) {
    BamFileView hf_view{fp_bam, fp_bai, fp_header};

    phase_return_t phasing_result;
    if (use_dvr_for_phasing) {
        phasing_result = kadayashi_dvr_single_region_wrapper1(
                fp_bam, fp_bai, fp_header, fai, ref_name, ref_start, ref_end,
                disable_interval_expansion, min_base_quality, min_varcall_coverage,
                min_varcall_fraction, max_clipping, min_strand_cov, min_strand_cov_frac,
                max_gapcompressed_seqdiv);
    } else {
        phasing_result = kadayashi_simple_single_region_wrapper1(
                fp_bam, fp_bai, fp_header, fai, ref_name, ref_start, ref_end,
                disable_interval_expansion, min_base_quality, min_varcall_coverage,
                min_varcall_fraction, max_clipping, min_strand_cov, min_strand_cov_frac,
                max_gapcompressed_seqdiv);
    }

    const pileup_pars_t pp_phased_round{.min_base_quality = min_base_quality,
                                        .min_varcall_coverage = min_varcall_coverage,
                                        .min_varcall_fraction = min_varcall_fraction,
                                        .max_clipping = max_clipping,
                                        .min_strand_cov = min_strand_cov,
                                        .min_strand_cov_frac = min_strand_cov_frac,
                                        .max_gapcompressed_seqdiv = max_gapcompressed_seqdiv,
                                        .retain_het_only = false,
                                        .retain_SNP_only = false,
                                        .use_bloomfilter = false,
                                        .disable_low_complexity_masking = false,
                                        .disable_region_expansion = true};
    chunk_t ck = variant_pileup_ht(hf_view, {}, fai, &phasing_result.qname2hp, ref_name, ref_start,
                                   ref_end, pp_phased_round);

    // variant phasing+genotype
    ck_derive_variant_genophase_from_phased_read(ck);

    // produce variants (no phaseblock info)
    ck_and_varcall_result_t ret;
    varcall_result_internal_t &vr = ret.vr;
    const std::string span_s = create_region_string(ref_name, ref_start, ref_end);
    const std::string refseq_s = kadayashi::hts_utils::fetch_seq(fai, span_s);

    for (const ta_t &varcall : ck.varcalls) {
        const auto var = derive_variant_fullinfo_from_varcall(varcall, refseq_s, ref_start, true);
        if (var.is_valid) {
            vr.variants.push_back(var);
        }
    }
    fix_variant_fullinfo_genotype_snp_in_del(vr.variants);

    vr.qname2hp =
            std::move(phasing_result.qname2hp);  // 0-index because this is an internal function
    vr.phasing_breakpoints = std::move(phasing_result.phasing_breakpoints);
    ret.ck = std::move(ck);
    return ret;
}
varcall_result_t kadayashi_phase_and_varcall_wrapper(samFile *fp_bam,
                                                     hts_idx_t *fp_bai,
                                                     sam_hdr_t *fp_header,
                                                     const faidx_t *fai,
                                                     const std::string_view ref_name,
                                                     const uint32_t ref_start,
                                                     const uint32_t ref_end,
                                                     const bool disable_interval_expansion,
                                                     const int min_base_quality,
                                                     const int min_varcall_coverage,
                                                     const float min_varcall_fraction,
                                                     const int max_clipping,
                                                     const int min_strand_cov,
                                                     const float min_strand_cov_frac,
                                                     const float max_gapcompressed_seqdiv,
                                                     const bool use_dvr_for_phasing) {
    ck_and_varcall_result_t ck_and_vr = kadayashi_phase_and_varcall(
            fp_bam, fp_bai, fp_header, fai, ref_name, ref_start, ref_end,
            disable_interval_expansion, min_base_quality, min_varcall_coverage,
            min_varcall_fraction, max_clipping, min_strand_cov, min_strand_cov_frac,
            max_gapcompressed_seqdiv, use_dvr_for_phasing);
    chunk_t &ck = ck_and_vr.ck;

    if (!ck.is_valid || ck.varcalls.empty()) {
        return {.qname2hp = std::move(ck_and_vr.vr.qname2hp), .variants = {}};
    }

    varcall_result_t ret{.qname2hp = ck_and_vr.vr.qname2hp,
                         .variants = {},
                         .phasing_breakpoints = ck_and_vr.vr.phasing_breakpoints};

    for (const auto &var : ck_and_vr.vr.variants) {  // return variants in dorado style
        if (var.is_valid) {
            ret.variants.push_back(convert_fullinfo_var_to_dorado_style(var));
        }
    }

    return ret;
}

}  // namespace kadayashi
