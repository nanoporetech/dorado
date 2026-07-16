#include "secondary/kadayashi/local_haplotagging.h"

#include "hts_utils/bam_utils.h"
#include "hts_utils/hts_types.h"
#include "secondary/consensus/variant_calling.h"
#include "secondary/kadayashi/bam_record_parsing.h"
#include "secondary/kadayashi/blocked_bloom_filter.h"
#include "secondary/kadayashi/faidx_utils.h"
#include "secondary/kadayashi/sequence_utility.h"
#include "string_utils.h"
#include "utils/container_utils.h"
#include "variant_graph.h"

#include <htslib/faidx.h>
#include <htslib/hts.h>
#include <htslib/kfunc.h>
#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <ostream>
#include <span>
#include <stdexcept>
#include <unordered_set>
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

constexpr char SENTINEL_REF_ALLELE[] = "!";
constexpr int SENTINEL_REF_ALLELE_L = 1;

constexpr int DORADO_FEATURE_MAT_READ_SENTINAL_LEN = 5;

constexpr int MEDAKA_FEATURE_MATRIX_MAX_VAR_LEN = 16777216;  // 1<<24

namespace {

bool check_ambig_in_string(const std::string_view &seq) {
    for (char c : seq) {
        const unsigned char lookup = md_op_table[static_cast<uint8_t>(c)];
        if (lookup != 2) {
            return true;
        }
    }
    return false;
}

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
            std::vector<uint8_t> &a = dst.back().allele;
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
                if ((dists[i][j] == dists[i][j - 1]) && (dists[i][j] == dists[i][j - 2])) {
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
                    } else if (i + 2 == (std::ssize(idx[i_mer]) - 1)) {
                        if ((dists[i_mer][i + 1] == d) || (dists[i_mer][i + 2] == d)) {
                            poss.push_back(static_cast<uint32_t>(i));
                            ok = 1;
                        } else {
                            ok = 0;
                        }
                    } else if (i + 1 == (std::ssize(idx[i_mer]) - 1)) {
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
                        "[kdys::{}] failed parsing mers (failed={}) or intervals_buf is empty "
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
interval_t expand_query_interval(dorado::secondary::BamFileView &hf,
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

    dorado::BamPtr aln(bam_init1());
    dorado::secondary::HtsItrPtr bamitr(sam_itr_querys(hf.idx, hf.hdr, itvl_left.c_str()));

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
    bamitr = dorado::secondary::HtsItrPtr(sam_itr_querys(hf.idx, hf.hdr, itvl_right.c_str()));
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
        for (const vc_allele_t &allele : var.alleles) {
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
                        "[kdys::{}] let pos {} set to unsure although flag is bad, due to its "
                        "many alleles & sufficient coverage in alt ({}/{})",
                        __func__, pos, tot_cov_alt, tot_cov_any);
            }
            var.is_accepted = FLAG_VARSTAT_UNSURE;
            var.type = FLAG_VAR_NA;
            return true;
        } else {
            if constexpr (DEBUG_LOCAL_HAPLOTAGGING) {
                LOG_DEBUG(
                        "[kdys::{}] straight ignore pos {} (flag is bad; size is {}, tot cov is "
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
        if ((var.alleles[0].allele[0] == SENTINEL_REF_ALLELE_INT) &&
            (var.alleles[1].allele[0] == SENTINEL_REF_ALLELE_INT)) {
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
                                "[kdys::{}] set pos {} as unsure (due to unphased) "
                                "tot_all_non_refs={} "
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
                    "[kdys::{}] pos {} in prefiltering fisher: phased-only, twosided p: {:.4f} ({} "
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
                const float hap0_tot =
                        std::accumulate(var.alleles.begin(), var.alleles.end(), 0.0f,
                                        [](float acc, const vc_allele_t &allele) -> float {
                                            return acc + (float)allele.cov.cov_hap0;
                                        });
                const float hap1_tot =
                        std::accumulate(var.alleles.begin(), var.alleles.end(), 0.0f,
                                        [](float acc, const vc_allele_t &allele) -> float {
                                            return acc + (float)allele.cov.cov_hap1;
                                        });
                const float tot_cov2 = std::accumulate(
                        var.alleles.begin(), var.alleles.end(), 0.0f,
                        [](float acc, const vc_allele_t &allele) -> float {
                            return acc + (float)allele.cov.cov_hap0 + (float)allele.cov.cov_hap1 +
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
                                "[kdys::{}] set pos {} as hom (2c; allel0all={}, tot_cov2={}, "
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
                            "[kdys::{}] pos {} was set to accepted hom, but flipping this to "
                            "unsure "
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
                            "[kdys::{}] set pos {} unsure (1a: weird coverage (top1={} top2={}; "
                            "all "
                            "alts={}))",
                            __func__, pos, top_cov_1, top_cov_2, tot_all_non_refs);
                }
            } else {
                var.is_accepted = FLAG_VARSTAT_REJECTED;
                if constexpr (DEBUG_LOCAL_HAPLOTAGGING) {
                    LOG_DEBUG(
                            "[kdys::{}] reject pos {} (1a: weird coverage ({} alleles) and they "
                            "are "
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

enum class var_classify_code {
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
    for (vc_allele_t &allele : var.alleles) {
        allele.cov.cov_tot_phased = allele.cov.cov_hap0 + allele.cov.cov_hap1;
        allele.cov.cov_tot_all = allele.cov.cov_tot_phased + allele.cov.cov_unphased;
    }
}
cov_extended_t variant_get_all_allele_coverage(const vc_variants1_val_t &var) {
    cov_extended_t ret{};

    for (const vc_allele_t &allele : var.alleles) {
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

void set_variant_depth(ta_t &var, const int depth) {
    var.min_depth = depth;
    var.median_depth = depth;
}

int get_ref_depth(const vc_variants1_val_t &var) {
    int ref_depth = 0;
    for (const vc_allele_t &allele : var.alleles) {
        if (std::empty(allele.allele)) {
            continue;
        }
        if ((allele.allele[0] == SENTINEL_REF_ALLELE_INT) && (allele.allele.back() == VAR_OP_X)) {
            ref_depth += allele.cov.cov_tot_all;
        }
    }
    return ref_depth;
}

int get_allele_depth_or_zero(const std::vector<int> &allele_depths, const size_t idx) {
    return (idx < std::size(allele_depths)) ? allele_depths[idx] : 0;
}

void set_merged_variant_depth(ta_t &var, std::vector<int> depths) {
    assert(!depths.empty());
    std::sort(depths.begin(), depths.end());

    var.min_depth = depths.front();
    const size_t mid = depths.size() / 2;
    if ((depths.size() % 2) == 0) {
        var.median_depth = (depths[mid - 1] + depths[mid]) / 2;
    } else {
        var.median_depth = depths[mid];
    }
}

uint32_t get_expected_allele_idx_for_haplotype(const ta_t &var, const uint32_t hap_idx) {
    assert(hap_idx < 2);

    if (var.type == TA_TYPE_HOM) {
        return 0;
    }

    const char gt = var.genotype[hap_idx * 2];
    assert((gt >= '0') && (gt <= '2'));
    if (var.type == TA_TYPE_HET) {
        const uint32_t ref_idx = (var.alleles[0][0] == SENTINEL_REF_ALLELE_INT) ? 0U : 1U;
        const uint32_t alt_idx = (ref_idx == 0) ? 1U : 0U;
        return (gt == '0') ? ref_idx : alt_idx;
    }

    return static_cast<uint32_t>(gt - '0');
}

int get_ref_allele_idx(const ta_t &var) {
    for (size_t i = 0; i < std::size(var.alleles); ++i) {
        if ((!std::empty(var.alleles[i])) && (var.alleles[i][0] == SENTINEL_REF_ALLELE_INT)) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

bool read_supports_haplotype_across_variants(const read_t &read,
                                             const chunk_t &ck,
                                             std::span<const uint32_t> var_indices,
                                             const uint32_t hap_idx) {
    size_t i_read_var = 0;
    for (const uint32_t var_idx : var_indices) {
        while ((i_read_var < std::size(read.vars)) && (read.vars[i_read_var].var_idx < var_idx)) {
            ++i_read_var;
        }
        if ((i_read_var >= std::size(read.vars)) || (read.vars[i_read_var].var_idx != var_idx)) {
            return false;
        }
        const uint32_t expected_allele_idx =
                get_expected_allele_idx_for_haplotype(ck.varcalls[var_idx], hap_idx);
        if (read.vars[i_read_var].allele_idx != expected_allele_idx) {
            return false;
        }
    }
    return true;
}

bool read_supports_reference_across_variants(const read_t &read,
                                             const chunk_t &ck,
                                             std::span<const uint32_t> var_indices) {
    size_t i_read_var = 0;
    for (const uint32_t var_idx : var_indices) {
        while ((i_read_var < std::size(read.vars)) && (read.vars[i_read_var].var_idx < var_idx)) {
            ++i_read_var;
        }
        if ((i_read_var >= std::size(read.vars)) || (read.vars[i_read_var].var_idx != var_idx)) {
            return false;
        }
        const int ref_allele_idx = get_ref_allele_idx(ck.varcalls[var_idx]);
        if ((ref_allele_idx < 0) ||
            (read.vars[i_read_var].allele_idx != static_cast<uint32_t>(ref_allele_idx))) {
            return false;
        }
    }
    return true;
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
    ret.code = var_classify_code::UNSET;

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
    enum class which_hap { HAP0, HAP1 };
    auto allele_dominates_haplotype = [&varcov](const vc_allele_t &allele, which_hap hap) {
        int a_cov_hap;
        int tot_cov_hap;
        if (hap == which_hap::HAP0) {
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
        const bool dominates_hap0 = allele_dominates_haplotype(a, which_hap::HAP0) &&
                                    !allele_dominates_haplotype(a, which_hap::HAP1);
        const bool dominates_hap1 = allele_dominates_haplotype(a, which_hap::HAP1) &&
                                    !allele_dominates_haplotype(a, which_hap::HAP0);
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
            ret.code = var_classify_code::HET_FISHER_COV_STRAND;
        } else {
            ret.is_accepted = FLAG_VARSTAT_UNSURE;
            ret.code = var_classify_code::HET_FISHER;
        }
    } else {
        if (ok_weaker) {
            ret.is_accepted = FLAG_VARSTAT_UNSURE;
            ret.code = var_classify_code::HET_WEAKERFISHER;
        } else {
            ret.is_accepted = FLAG_VARSTAT_REJECTED;
            ret.code = var_classify_code::HET_NOTHING;
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
            ret.code = var_classify_code::REJECT_TOO_MANY_ALLELE;
        }
        has_decided = true;
    }

    // not clean het, and top two are ref(substitution) and ref(del), ignore
    if (!has_decided && (var.alleles[0].allele[0] == SENTINEL_REF_ALLELE_INT) &&
        (var.alleles[1].allele[0] == SENTINEL_REF_ALLELE_INT)) {
        ret.is_accepted = FLAG_VARSTAT_REJECTED;
        ret.type = FLAG_VAR_NA;
        ret.code = var_classify_code::REJECT_REFREF;
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
                ret.code = var_classify_code::HOM_STRAND;
            } else {
                ret.is_accepted = FLAG_VARSTAT_UNSURE;
                ret.code = var_classify_code::HOM_NOTHING;
            }
            has_decided = true;
        }
    }
    if (!has_decided) {
        for (const vc_allele_t &allele : var.alleles) {
            if (allele.allele[0] != SENTINEL_REF_ALLELE_INT &&
                static_cast<float>(allele.cov.cov_tot_all) >= 0.3f * varcov.cov.cov_tot_all) {
                ret.is_accepted = FLAG_VARSTAT_UNSURE;
                ret.type = FLAG_VAR_NA;
                ret.code = var_classify_code::HOM_RESUCEFAIL;
                has_decided = true;
                break;
            }
        }
    }

    if (!has_decided) {
        if (varcov.n_long_alts >= N_LONG_ALTS) {  // long alt
            ret.is_accepted = FLAG_VARSTAT_UNSURE;
            ret.code = var_classify_code::SUS_LONG_ALT;
        } else if (var.alleles.size() > 4) {  // many alts; 4:ref/ref/alt/alt
            int n_allele_with_suf_cov = 0;
            for (const vc_allele_t &allele : var.alleles) {
                if (allele.allele[0] != SENTINEL_REF_ALLELE_INT &&
                    allele.cov.cov_tot_all >= MIN_SUF_COV_ALLELE) {
                    n_allele_with_suf_cov++;
                }
            }
            if (n_allele_with_suf_cov >= 2) {
                ret.is_accepted = FLAG_VARSTAT_UNSURE;
                ret.code = var_classify_code::SUS_MULTI_ALLELE;
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
            (!((a0.allele.back() == VAR_OP_X) && (a0.allele[0] == SENTINEL_REF_ALLELE_INT)) ||
             !((a1.allele.back() != VAR_OP_X) && (a1.allele[0] == SENTINEL_REF_ALLELE_INT)));
    const int is_rejected_non_ref = is_non_ref && (var.is_accepted == FLAG_VARSTAT_REJECTED);
    int is_indel = 0;
    int n_long_indel = 0;
    if (var.alleles.size() > 1) {
        is_indel = ((a0.allele.back() == VAR_OP_I) || (a1.allele.back() == VAR_OP_I) ||
                    (a0.allele.back() == VAR_OP_D) || (a1.allele.back() == VAR_OP_D));
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
                ret.code = var_classify_code::SUS_REPETITIVE;
            }
        }
    }
    if ((ret.is_accepted == FLAG_VARSTAT_ACCEPTED) &&
        (varcov.cov.cov_tot_all - varcov.cov.cov_tot_phased) >
                (varcov.cov.cov_hap0 + varcov.cov.cov_hap1)) {
        ret.is_accepted = FLAG_VARSTAT_UNSURE;
        ret.code = var_classify_code::SUS_PHASE;
    }

    if constexpr (DEBUG_LOCAL_HAPLOTAGGING) {
        LOG_DEBUG(
                "[kdys::{}] pos={} is_accepted={} type={} code={}; fisher_p={}; is_at_repeat={}; "
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

phase_return_t kadayashi_simple_single_region_wrapper1(samFile *fp_bam,
                                                       hts_idx_t *fp_bai,
                                                       sam_hdr_t *fp_header,
                                                       const faidx_t *fai,
                                                       const std::string_view ref_name,
                                                       const uint32_t ref_start,
                                                       const uint32_t ref_end,
                                                       const std::string_view readgroup,
                                                       const bool disable_interval_expansion,
                                                       const int min_base_quality,
                                                       const int min_varcall_coverage,
                                                       const float min_varcall_fraction,
                                                       const int max_clipping,
                                                       const int min_strand_cov,
                                                       const float min_strand_cov_frac,
                                                       const float max_gapcompressed_seqdiv) {
    const pileup_pars_t pp = {
            .readgroup = std::string(readgroup),
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

struct ta_t_del_info {
    int del_len;
    int del_idx;
};
ta_t_del_info ta_t_check_and_get_del_allele(const ta_t &var) {
    // (note: if both are deletions, return the info of the longer one)
    int len = 0;
    int idx = 0;
    for (size_t i = 0; i < var.alleles.size(); i++) {
        const std::vector<uint8_t> &allele = var.alleles[i];
        int allele_len = static_cast<int>(allele.size()) - 1;
        if (allele.back() == VAR_OP_D && allele_len > len) {
            len = allele_len;
            idx = i;
        }
    }
    return ta_t_del_info{.del_len = len, .del_idx = idx};
}

std::pair<std::string, std::string> ta_t_reconstruct_haplotypes_for_del_absorbtion(
        chunk_t &ck,
        uint32_t range_start,  // inclusive
        uint32_t range_end,    // exclusive
        const std::string &ref_seq_substring,
        const std::vector<uint32_t> &var_indices) {
    // Helper routine to ck_absorb_variants_covered_by_deletion_run_on_other_hap.
    // Given a range and variants selected within the range,
    // produce the full sequence of the haplotypes.
    // (For simplicity the reconstruction is done by first retrieving
    //  the refseq, then apply variants one by one in reverse order.
    //  This function is not expected to be called on large regions so this
    //  should not matter much.)
    // `var_indices`:
    //  - must be sorted
    //  - all variants refered must be either hom or phased het
    //  - the first var is also assumed to be/have a deletion

    if (var_indices.empty()) {
        return {"", ""};
    }

    std::string hap0 =
            ref_seq_substring.substr(range_start - ck.abs_start, range_end - range_start);
    std::string hap1 = hap0;

    auto insert_one_var_to_hap_recon = [&range_start](std::string &hap, const ta_t &var,
                                                      uint32_t i_allele) {
        const uint8_t cigar_op = var.alleles[i_allele].back();
        const uint32_t cigar_len = var.alleles[i_allele].size() - 1;
        const std::string var_str = nt4seq2seq(var.alleles[i_allele]);
        if (cigar_op == VAR_OP_X) {
            hap[var.pos - range_start] = var_str[0];
        } else if (cigar_op == VAR_OP_I) {  // +1 because pos considered refbase
            hap = hap.substr(0, var.pos + 1 - range_start) + var_str +
                  hap.substr(var.pos + 1 - range_start);
        } else if (cigar_op == VAR_OP_D) {  // +1 because pos considered refbase
            hap = hap.substr(0, var.pos + 1 - range_start) +
                  ((var.pos + 1 + cigar_len - range_start >= hap.size())
                           ? ""
                           : hap.substr(var.pos + 1 + cigar_len - range_start));
        } else {
            throw std::runtime_error{
                    "[kdys::ta_t_reconstruct_haplotypes_for_del_absorbtion] unexpected cigar op: " +
                    std::to_string(cigar_op)};
        }
    };

    for (int i_idx = static_cast<int>(var_indices.size() - 1); i_idx >= 0; i_idx--) {
        const uint32_t var_idx = var_indices[i_idx];
        const ta_t &var = ck.varcalls[var_idx];

        if (var.genotype[0] == var.genotype[2]) {  // hom
            insert_one_var_to_hap_recon(hap0, var, 0);
            insert_one_var_to_hap_recon(hap1, var, 0);
        } else {  // het
            assert(var.genotype[1] == '|');
            if (var.type == TA_TYPE_HETMULTI) {
                if (var.genotype[0] == '1') {  // for chunk_t variants, multi-allelic
                                               // is not genotype 1/2 yet. fullinfo vars are.
                    assert(var.genotype[2] == '0');
                    insert_one_var_to_hap_recon(hap0, var, 1);
                    insert_one_var_to_hap_recon(hap1, var, 0);
                } else if (var.genotype[0] == '0') {
                    assert(var.genotype[2] == '1');
                    insert_one_var_to_hap_recon(hap0, var, 0);
                    insert_one_var_to_hap_recon(hap1, var, 1);
                } else {
                    throw std::runtime_error{
                            std::string("[kdys::") + __func__ +
                            "] unexpected genotype(multi-allelic): " + var.genotype.data() +
                            " , query range was " + ck.refname + ":" + std::to_string(range_start) +
                            "-" + std::to_string(range_end) + " , var.pos was " +
                            std::to_string(var.pos)};
                }
            } else {
                if (var.genotype[0] == '1') {
                    assert(var.genotype[2] == '0');
                    insert_one_var_to_hap_recon(
                            hap0, var, var.alleles[0][0] == SENTINEL_REF_ALLELE_INT ? 1 : 0);
                } else if (var.genotype[2] == '1') {
                    assert(var.genotype[0] == '0');
                    insert_one_var_to_hap_recon(
                            hap1, var, var.alleles[0][0] == SENTINEL_REF_ALLELE_INT ? 1 : 0);
                } else {
                    throw std::runtime_error{
                            std::string("[kdys::") + __func__ +
                            "] unexpected genotype(single-allelic): " + var.genotype.data() +
                            " , query range was " + ck.refname + ":" + std::to_string(range_start) +
                            "-" + std::to_string(range_end) + " , var.pos was " +
                            std::to_string(var.pos)};
                }
            }
        }  // insert var seq into haplotype seq for one var
    }  // insert var seqs

    return {std::move(hap0), std::move(hap1)};
}

void ck_absorb_variants_covered_by_deletion_run_on_other_hap(
        chunk_t &ck,
        const std::unordered_map<uint32_t, uint8_t> &phasing_breakpoints,
        const std::string &ref_seq_substring) {
    // Consider the following dummy case:
    //  pos   123456789012
    //  ref   AAAAAAAAAATT
    //  hap1  AA--------TT
    //  hap2  AAAGGAAGAATT
    //                 ^ins(G)
    // We want to report this as one multi-allelic variant:
    //  chrom 2 AAAAAAAAA A,AAGGAAGAGA 1|2
    // However, if any variant within a proposed range
    //  is unconfident, we will not try to merge them. Instead,
    //  all variants within the range are marked as unconfident.
    //
    // This should happen after we have produced genotype & phasing string,
    //  and will only treat confident & phased variants. (If there are
    //  unsure variants within an affected region, the reigon will be
    //  handed to the big model & all kadayashi variants replaced anyways;
    //  unphased het should be always unconfident.)
    //
    // Note that in the example above, if hap2 has a deletion that
    //  overlaps with and outruns hap1's deletion, we must look beyond
    //  hap1's deletion to see if any variant need to be included. Repeat
    //  until both are in match with the ref.
    // Access to the reference sequence is required because
    //  ref bases in between of variants were not saved in
    //  the chunk struct.
    for (size_t i_var = 0; i_var < ck.varcalls.size(); i_var++) {
        ta_t &var = ck.varcalls[i_var];

        if (var.is_used != TA_STAT_ACCEPTED || var.genotype[1] != '|') {
            continue;
        }

        const ta_t_del_info stat1 = ta_t_check_and_get_del_allele(var);
        if (stat1.del_len == 0) {
            continue;
        }

        // We will get the range first, and then reconstruct
        // the two haplotypes, before marking varints as
        // unused and replace one of them with the generated sequence.

        // (get range)
        uint32_t range_start = var.pos;
        uint32_t range_end = var.pos + 1 + stat1.del_len;  // +1 because pos considered the refbase
        std::vector<uint32_t> var_indices;
        var_indices.push_back(i_var);
        bool range_is_clean = true;
        for (size_t j_var = i_var + 1; j_var < ck.varcalls.size(); j_var++) {
            const ta_t &var2 = ck.varcalls[j_var];
            if (var2.pos >= range_end) {
                break;
            }

            if (var2.is_used != TA_STAT_ACCEPTED) {
                range_is_clean = false;
            }

            if (!(var2.genotype[1] == '|' ||
                  var2.genotype[0] == var.genotype[2])) {  // require hom or phased het
                range_is_clean = false;
            }

            var_indices.push_back(j_var);

            const ta_t_del_info stat2 = ta_t_check_and_get_del_allele(var2);
            if (stat2.del_len != 0) {
                const uint32_t tmp_end = var2.pos + stat2.del_len + 1;
                if (tmp_end > range_end) {
                    range_end = tmp_end;
                }
            }
        }
        if (var_indices.size() <= 1) {  // nothing to be absorbed
            continue;
        }

        // (if there's phasing breakpoints in the range, don't do anything)
        for (uint32_t i = 0; i < var_indices.size() - 1; i++) {
            // [0,size-1) because breakpoint are set on a variant position
            // when phasing to the next variant is unknown.
            if (phasing_breakpoints.find(ck.varcalls[var_indices[i]].pos) !=
                phasing_breakpoints.end()) {
                range_is_clean = false;
                break;
            }
        }

        // (reconstruct haplotypes, or mark everything as unconfident)
        if (range_is_clean) {
            std::pair<std::string, std::string> hap_recon =
                    ta_t_reconstruct_haplotypes_for_del_absorbtion(ck, range_start, range_end,
                                                                   ref_seq_substring, var_indices);

            // (replace the original: now phased multi-allelic)
            // (the haplotype reconstruction's phasing order was in
            // concordance with the first variant)
            // caveat: `allele2readIDs` is not updated.
            //  We only consolidate variants when they are
            //  coverved by deletion(s) on the other hap, this
            //  should rarely exceed read boundary. And since genophase
            //  is already collected, `allele2readIDs` will be used
            //  by nobody.
            std::array<int, 2> spanning_allele_depths{0, 0};
            int spanning_ref_depth = 0;
            {
                std::span<const uint32_t> var_indices_span{var_indices};
                for (const read_t &read : ck.reads) {
                    if ((read.start_pos > range_start) || (read.end_pos < range_end)) {
                        continue;
                    }
                    if (read_supports_reference_across_variants(read, ck, var_indices_span)) {
                        ++spanning_ref_depth;
                    }
                    for (uint32_t hap_idx = 0; hap_idx < std::size(spanning_allele_depths);
                         ++hap_idx) {
                        if (read_supports_haplotype_across_variants(read, ck, var_indices_span,
                                                                    hap_idx)) {
                            ++spanning_allele_depths[hap_idx];
                        }
                    }
                }
            }
            if (var.type != TA_TYPE_HETMULTI) {
                var.genotype[0] += 1;
                var.genotype[2] += 1;
            }
            std::vector<int> merged_depths;
            merged_depths.reserve(var_indices.size());
            for (const uint32_t var_idx : var_indices) {
                merged_depths.push_back(ck.varcalls[var_idx].median_depth);
            }
            set_merged_variant_depth(var, std::move(merged_depths));
            var.ref_depth = spanning_ref_depth;
            var.allele_depths = {spanning_allele_depths[0], spanning_allele_depths[1]};
            var.type = TA_TYPE_HETMULTI_CONSOLIDATED;
            var.ref_len = range_end - range_start;
            var.alleles[0] = seq2nt4seq(hap_recon.first);
            var.alleles[0].push_back(VAR_OP_CONSOLIDATED_MULTIALLELE);
            var.alleles[1] = seq2nt4seq(hap_recon.second);
            var.alleles[1].push_back(VAR_OP_CONSOLIDATED_MULTIALLELE);

            // (mark absorbed vars as not used)
            for (uint32_t i = 1; i < var_indices.size(); i++) {
                ck.varcalls[var_indices[i]].is_used = TA_STAT_UNCALLED;
            }
        } else {  // range not clean, mark unconf
            for (uint32_t i = 0; i < var_indices.size(); i++) {
                ck.varcalls[var_indices[i]].is_used = TA_STAT_UNSURE;
                ck.varcalls[var_indices[i]].genotype[1] = '/';
            }
        }
    }  // iter through all called variants
}

void ck_derive_variant_genophase_from_phased_read(chunk_t &ck) {
    // note: for multi-allele variant, the genotype will be assigned
    //  0 or 1 here rather than 1 or 2. Afterwards
    //  derive_variant_fullinfo_from_varcall is expected to be called.
    // note2: we don't want to impose any coverage check here. That
    //  is up to the variant classifier.
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

            // note: here set both het and multihet to 01, let this be resolved
            // when we parse ta_t var into fullinfo var
            if (n_hap0 > n_hap1) {
                var.genotype = {'0', '|', '1', '\0'};
            } else if (n_hap1 > n_hap0) {
                var.genotype = {'1', '|', '0', '\0'};
            } else {  // unphased
                var.genotype[0] = '0';
                var.genotype[2] = '1';
            }
        }  // determine het variant phase+geno

    }  // iter through variants
}

variant_fullinfo_t derive_variant_fullinfo_from_varcall(const ta_t &var,
                                                        const std::string_view refseq_substring,
                                                        const uint32_t ref_start,
                                                        const bool ambig_ref) {
    // Note:
    //   - Variant position will be in 0-index.
    //   - Multi-allele variant should have genotypes 1 or 2.
    variant_fullinfo_t ret;
    ret.ref_depth = var.ref_depth;
    ret.min_depth = var.min_depth;
    ret.median_depth = var.median_depth;

    std::string ref_s;
    std::string ref_consolidatedmultiallel_s;
    std::string alt_s;
    std::string alt2_s;

    // store ref base
    if (var.type == TA_TYPE_HETMULTI_CONSOLIDATED) {
        ref_consolidatedmultiallel_s =
                std::string(refseq_substring.substr(var.pos - ref_start, var.ref_len));
        if ((!ambig_ref) && check_ambig_in_string(ref_consolidatedmultiallel_s)) {
            ret.is_valid = false;
            ret.is_confident = false;
            return ret;
        }
    } else {
        if (var.pos < ref_start) {  // might happen if pileup used expanded interval
            ret.is_valid = false;
            ret.is_confident = false;
            return ret;
        } else {
            const int tmppos = var.pos - ref_start;  // 0-index
            ref_s += refseq_substring[tmppos];
            if ((!ambig_ref) && check_ambig_in_string(ref_s)) {
                ret.is_valid = false;
                ret.is_confident = false;
                return ret;
            }
        }
    }

    // store alt base
    int is_del0 = 0;
    int is_ins0 = 0;
    int is_del1 = 0;
    int is_ins1 = 0;
    if (var.type == TA_TYPE_HOM) {
        if (var.alleles[0][0] != SENTINEL_REF_ALLELE_INT) {
            alt_s = nt4seq2seq(var.alleles[0]);
            ret.alt_depth0 = get_allele_depth_or_zero(var.allele_depths, 0);
            is_del0 = var.alleles[0].back() == VAR_OP_D;
            is_ins0 = var.alleles[0].back() == VAR_OP_I;
        } else {
            assert(var.alleles.size() > 1);
            alt_s = nt4seq2seq(var.alleles[1]);
            ret.alt_depth0 = get_allele_depth_or_zero(var.allele_depths, 1);
            is_del0 = var.alleles[1].back() == VAR_OP_D;
            is_ins0 = var.alleles[1].back() == VAR_OP_I;
        }
    } else {
        if (var.alleles[0][0] != SENTINEL_REF_ALLELE_INT &&
            var.alleles[1][0] != SENTINEL_REF_ALLELE_INT) {  // multi-allele, need to write both
            alt_s = nt4seq2seq(var.alleles[0]);
            alt2_s = nt4seq2seq(var.alleles[1]);
            ret.alt_depth0 = get_allele_depth_or_zero(var.allele_depths, 0);
            ret.alt_depth1 = get_allele_depth_or_zero(var.allele_depths, 1);
            is_del0 = var.alleles[0].back() == VAR_OP_D;
            is_ins0 = var.alleles[0].back() == VAR_OP_I;
            if (var.alleles.size() > 1) {
                is_del1 = var.alleles[1].back() == VAR_OP_D;
                is_ins1 = var.alleles[1].back() == VAR_OP_I;
            }
        } else {  // (regular case)
            if (var.alleles[0][0] != SENTINEL_REF_ALLELE_INT) {
                alt_s = nt4seq2seq(var.alleles[0]);
                ret.alt_depth0 = get_allele_depth_or_zero(var.allele_depths, 0);
                is_del0 = var.alleles[0].back() == VAR_OP_D;
                is_ins0 = var.alleles[0].back() == VAR_OP_I;
            } else {
                alt_s = nt4seq2seq(var.alleles[1]);
                ret.alt_depth0 = get_allele_depth_or_zero(var.allele_depths, 1);
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

    ret.pos0 = var.pos;
    ret.pos1 = 0;
    ret.qual1 = 0;
    if (var.type == TA_TYPE_HETMULTI_CONSOLIDATED) {
        ret.ref_allele_seq0 = ref_consolidatedmultiallel_s;
        ret.alt_allele_seq0 = alt_s;
    } else {
        if (is_del0) {
            ret.ref_allele_seq0 = ref_s[0] + alt_s;
            ret.alt_allele_seq0 = ref_s[0];
        } else if (is_ins0) {
            ret.ref_allele_seq0 = ref_s[0];
            ret.alt_allele_seq0 = ref_s[0] + alt_s;
        } else {
            ret.ref_allele_seq0 = ref_s;
            ret.alt_allele_seq0 = alt_s;
        }
    }
    ret.genotype0[0] = var.genotype[0];
    ret.genotype0[1] = var.genotype[1];
    ret.genotype0[2] = var.genotype[2];
    ret.is_phased0 = var.genotype[1] == '|';

    ret.is_multi_allele =
            (var.type == TA_TYPE_HETMULTI || var.type == TA_TYPE_HETMULTI_CONSOLIDATED);
    if (ret.is_multi_allele) {
        ret.pos1 = var.pos;
        if (var.type == TA_TYPE_HETMULTI_CONSOLIDATED) {
            ret.ref_allele_seq1 = ref_consolidatedmultiallel_s;
            ret.alt_allele_seq1 = alt2_s;
        } else {
            if (is_del1) {
                ret.ref_allele_seq1 = ref_s[0] + alt2_s;
                ret.alt_allele_seq1 = ref_s[0];
            } else if (is_ins1) {
                ret.ref_allele_seq1 = ref_s[0];
                ret.alt_allele_seq1 = ref_s[0] + alt2_s;
            } else {
                ret.ref_allele_seq1 = ref_s;
                ret.alt_allele_seq1 = alt2_s;
            }
        }

        // adjust genotype from {0,1} to {1,2}
        ret.genotype0[0] += 1;
        ret.genotype0[2] += 1;

        // assigned genotype to the other allele
        ret.genotype1[0] = ret.genotype0[2];
        ret.genotype1[1] = ret.genotype0[1];
        ret.genotype1[2] = ret.genotype0[0];
        ret.is_phased1 = var.genotype[1] == '|';
    }

    if ((!ambig_ref) &&
        (check_ambig_in_string(ret.ref_allele_seq0) || check_ambig_in_string(ret.ref_allele_seq1) ||
         check_ambig_in_string(ret.alt_allele_seq0) ||
         check_ambig_in_string(ret.alt_allele_seq1))) {
        ret.is_valid = false;
        ret.is_confident = false;
    }

    return ret;
}

enum class read_downsampling_rw {
    READ_DOWNSAMPLING_QUERY_ONLY,
    READ_DOWNSAMPLING_QUERY_AND_UPDATE
};
bool read_downsampling_query_or_update_counter(read_downsampling_rw rw,
                                               uint32_t r_start_pos,
                                               uint32_t r_end_pos,
                                               uint32_t itvl_start,
                                               uint32_t itvl_end,
                                               std::vector<int> &downsample_counter,
                                               int downsample_window,
                                               int downsample_readcap) {
    // Used by pileup_ht and feature matrix generator.
    // If rw is set to READ_DOWNSAMPLING_QUERY_ONLY, return true when read
    //  can be accepted (i.e. not filtered out based on the downsample counter),
    //  false otherwise, and does not update the counter.
    // If rw is set to READ_DOWNSAMPLING_QUERY_AND_UPDATE, return under the same
    //  rules but the counter will be updated.
    // The query is cheap under the current heuristic, where we only check whether
    //  the first slot is overflowing or not. This works reasonably ok when
    //  downsample_window is small (e.g. 1000).

    assert(downsample_window > 0);

    bool read_may_be_accepted = true;

    const uint32_t effective_r_start = (r_start_pos > itvl_start ? r_start_pos - itvl_start : 0);
    const uint32_t iw_s = effective_r_start / downsample_window;

    if (downsample_counter[iw_s] < downsample_readcap) {  // update the counter of downsampling
        read_may_be_accepted = true;

        if (rw == read_downsampling_rw::READ_DOWNSAMPLING_QUERY_AND_UPDATE) {
            const uint32_t effective_r_end = (r_end_pos > itvl_end ? itvl_end : r_end_pos) -
                                             (r_start_pos > itvl_start ? r_start_pos : itvl_start);
            const size_t n_counter = downsample_counter.size();
            uint32_t iw_n = effective_r_end / downsample_window;
            iw_n = iw_n == 0 ? 1 : iw_n;
            iw_n = (iw_s + iw_n) > (uint32_t)n_counter ? n_counter - iw_s : iw_n;
            for (uint32_t tmpi = iw_s; tmpi < iw_s + iw_n; tmpi++) {
                downsample_counter[tmpi] = downsample_counter[tmpi] < INT_MAX
                                                   ? downsample_counter[tmpi] + 1
                                                   : downsample_counter[tmpi];
            }
        }
    } else {
        read_may_be_accepted = false;
    }
    return read_may_be_accepted;
}
}  // namespace

dorado::secondary::Variant convert_fullinfo_var_to_dorado_style(const variant_fullinfo_t &var,
                                                                const bool keep_phasing) {
    if (!var.is_valid) {
        return {};
    }
    std::pair<char, char> genotype;
    std::string_view ref = var.ref_allele_seq0;
    std::vector<std::string> alts;
    std::vector<int> alt_depths;
    int ref_depth = var.ref_depth;

    const auto make_gt_string = [&keep_phasing, &genotype](const bool is_phased) {
        const char delimiter = (keep_phasing && is_phased) ? '|' : '/';
        return std::string{genotype.first, delimiter, genotype.second};
    };

    if (var.is_multi_allele) {
        std::string alt0 = var.alt_allele_seq0;
        std::string alt1 = var.alt_allele_seq1;
        genotype = (var.genotype0[0] < var.genotype0[2]) ? std::pair<char, char>{'2', '1'}
                                                         : std::pair<char, char>{'1', '2'};
        if (var.ref_allele_seq0 != var.ref_allele_seq1) {
            if (var.ref_allele_seq0.size() > var.ref_allele_seq1.size()) {
                ref = var.ref_allele_seq0;
                alt1.append(ref.data() + std::size(var.ref_allele_seq1),
                            ref.size() - std::size(var.ref_allele_seq1));
            } else {
                ref = var.ref_allele_seq1;
                alt0.append(ref.data() + std::size(var.ref_allele_seq0),
                            ref.size() - std::size(var.ref_allele_seq0));
            }
        }
        // check that we haven't just filled one alt with the ref, if so then clean up
        if ((alt0 == ref) && (alt1 == ref)) {
            // we're in a completely broken state
            return {};
        } else if (alt0 == ref) {
            alts = {std::move(alt1)};
            alt_depths = {var.alt_depth1};
            ref_depth += var.alt_depth0;
            genotype = (var.genotype0[0] < var.genotype0[2]) ? std::pair<char, char>{'1', '0'}
                                                             : std::pair<char, char>{'0', '1'};
        } else if (alt1 == ref) {
            alts = {std::move(alt0)};
            alt_depths = {var.alt_depth0};
            ref_depth += var.alt_depth1;
            genotype = (var.genotype0[0] < var.genotype0[2]) ? std::pair<char, char>{'0', '1'}
                                                             : std::pair<char, char>{'1', '0'};
        } else {
            alts = {std::move(alt0), std::move(alt1)};
            alt_depths = {var.alt_depth0, var.alt_depth1};
        }
    } else {
        genotype = {var.genotype0[0], var.genotype0[2]};
        alts = {var.alt_allele_seq0};
        alt_depths = {var.alt_depth0};
    }
    if (!(var.is_phased0 && keep_phasing)) {
        if (genotype.first > genotype.second) {
            std::swap(genotype.first, genotype.second);
        }
        const bool needs_sort = !std::is_sorted(std::begin(alts), std::end(alts));
        if (needs_sort) {
            if (std::size(alts) == 2) {
                std::swap(alts[0], alts[1]);
                if (std::size(alt_depths) == 2) {
                    std::swap(alt_depths[0], alt_depths[1]);
                }
            } else {
                spdlog::debug(
                        "[convert_fullinfo_var_to_dorado_style] Not sorting variant record at {} "
                        "with {}!=2 alts",
                        var.pos0, std::size(alts));
            }
        }
    }

    const float qual = var.is_confident ? dorado::secondary::VCF_MAX_GQ_CAP : 0.0f;
    dorado::secondary::Variant ret{
            .seq_id = -1,  // placeholder
            .pos = var.pos0,
            .ref = std::string(ref),
            .alts = alts,
            .filters = {var.is_confident ? dorado::secondary::Filters::Pass
                                         : dorado::secondary::Filters::LowQual},
            .info = {},
            .qual = qual,
            .format = {{"GT", make_gt_string(var.is_phased0)},
                       {"GQ", dorado::secondary::gq_from_qual(qual)},
                       {"DP", std::to_string(var.median_depth)},
                       {dorado::secondary::FORMAT_TAG_VARIANT_SOURCE,
                        dorado::secondary::VARIANT_SOURCE_KADAYASHI},
                       {"AD",
                        std::to_string(ref_depth) + "," +
                                dorado::utils::print_container_as_string(alt_depths, ",", false)}},
            .rstart = 0,
            .rend = 0,
    };

    if ((var.median_depth != var.min_depth) && (ret.ref.length() > 1)) {
        ret.format["MIN_DP"] = std::to_string(var.min_depth);
    }
    return ret;
}

chunk_t variant_pileup_ht(dorado::secondary::BamFileView &hf,
                          const variants_t &ht_refvars,
                          const faidx_t *fai,
                          const str2int_t *qname2hp,  // 0-index
                          const std::string_view refname,
                          const uint32_t itvl_start,
                          const uint32_t itvl_end,
                          const pileup_pars_t &pp) {
    // A simpler pileup that allows hom variants, non-SNPs and multi-alleles.

    constexpr bool ENABLE_DOWNSAMPLE = true;

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
    if constexpr (DEBUG_LOCAL_HAPLOTAGGING) {
        LOG_DEBUG("[kdys::{}] pileup at {}:{}-{} (1-index, close-open)", __func__, refname,
                  ck.abs_start + 1, ck.abs_end);
    }

    // note: use `span_{start,end}` rather than `ck.abs_{start,end}`
    // except for when trimming off the interval expansion (if any).
    uint32_t span_start = ck.abs_start;
    uint32_t span_end = ck.abs_end;

    const std::string itvl = create_region_string(refname, ck.abs_start, ck.abs_end);
    dorado::secondary::HtsItrPtr bamitr(sam_itr_querys(hf.idx, hf.hdr, itvl.c_str()));
    dorado::BamPtr aln(bam_init1());

    if (!pp.disable_region_expansion) {
        interval_t new_itvl = expand_query_interval(hf, refname, ck.abs_start, ck.abs_end);
        span_start = new_itvl.start;
        span_end = new_itvl.end;
        const std::string itvl2 = create_region_string(refname, span_start, span_end);

        bamitr = dorado::secondary::HtsItrPtr(sam_itr_querys(hf.idx, hf.hdr, itvl2.c_str()));
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
    const int n_counter = (span_end - span_start) / DOWNSAMPLE_WINDOW + 1;
    if (ENABLE_DOWNSAMPLE) {
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

        const bool md_is_ok = sancheck_MD_tag_exists_and_is_valid(aln.get());
        if (!md_is_ok) {
            continue;
        }
        if (aln.get()->core.n_cigar == 0) {
            continue;
        }
        if (to_exclude_by_flags(aln.get(), 4 | 256 | 2048)) {
            continue;
        }
        if (to_exclude_by_low_mapq(aln.get(), pp.min_mapq)) {
            continue;
        }
        if (to_exlucde_by_high_de_tag(aln.get(), pp.max_gapcompressed_seqdiv)) {
            continue;
        }
        // filter by readgroup (adapted from medaka_bamiter.cpp)
        if (!pp.readgroup.empty()) {
            const uint8_t *rg = bam_aux_get(aln.get(), "RG");
            if (rg) {
                errno = 0;  // reset
                const char *rg_val = bam_aux2Z(rg);
                if (errno == EINVAL) {
                    continue;
                }
                if (pp.readgroup != rg_val) {
                    continue;
                }
            }
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

        if (ENABLE_DOWNSAMPLE) {
            const bool read_may_be_accepted = read_downsampling_query_or_update_counter(
                    read_downsampling_rw::READ_DOWNSAMPLING_QUERY_AND_UPDATE, r_start_pos,
                    r_end_pos, span_start, span_end, downsample_counter, DOWNSAMPLE_WINDOW,
                    DOWNSAMPLE_READCAP);
            if (!read_may_be_accepted) {
                downsample_filtered++;
                continue;  // go parse the next read
            }
        }

        n_reads++;
        // collect variants of the read
        read_t r{
                .start_pos = r_start_pos,
                .end_pos = r_end_pos,
                .ID = n_reads,
                .vars = {},
                .hp = hp,
                .votes_diploid = {0, 0},
                .strand = !!(aln.get()->core.flag & 16),
                .de = get_tag_de_f(aln.get()),
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
                if (r.vars[i].pos < span_start) {
                    continue;
                }
                if (r.vars[i].pos >= span_end) {
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
        spdlog::debug(
                "[kdys::{}] ht size is {}, ck size {} (downsample={}, filtered={}), n_reads={}",
                __func__, ht.size(), ck.reads.size(), ENABLE_DOWNSAMPLE, downsample_filtered,
                n_reads);
    }

    if (pileup_failed) {
        spdlog::error(
                "[kdys::{}] query {}:{}-{} (1-index [) ]) pileup's initial collection failed.",
                __func__, refname, ck.abs_start + 1, ck.abs_end);
        return {};
    }

    if (ck.qnames.size() != ck.reads.size()) {
        spdlog::error(
                "[kdys::{}] ck qnames and reads buffers have different lenghts ({} and {}), "
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
            hplowcmp_mask = get_lowcmp_mask(fai, refname, span_start, span_end);
        }
        const std::string span_s = create_region_string(refname, span_start, span_end);
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
            const int read_prev_var_was_del =
                    (j > 0 && (read.vars[j - 1].allele.back() == VAR_OP_D));
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

        const bool is_done = classify_variant_prefilter(q, pos, refseq_s.c_str(), refseq_l,
                                                        span_start, qname2hp);

        if (is_done) {
            continue;
        }

        q.is_accepted = FLAG_VARSTAT_REJECTED;  // default to reject
        q.type = FLAG_VAR_NA;
        if (qname2hp) {  // phased varcall
            assert(fai);
            const var_classify_t result =
                    classify_variant_phased(q, pos, refseq_s.c_str(), refseq_l, span_start,
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
                if ((q.alleles[0].allele[0] == SENTINEL_REF_ALLELE_INT) ||
                    (q.alleles[1].allele[0] == SENTINEL_REF_ALLELE_INT)) {
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
                        "[kdys::{}] unphased pileup pos {} : c0={} ({:c}) c1={} ({:c}) "
                        "ratio={:.2f} -> "
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
        const vc_variants1_val_t &q = ht[pos];
        const uint8_t var_stat = q.is_accepted;
        if ((var_stat & (FLAG_VARSTAT_ACCEPTED | FLAG_VARSTAT_UNSURE))) {
            if (pp.retain_het_only && q.type != FLAG_VAR_HET) {
                continue;
            }
            if (valid_poss.size() > 0 && (pos == valid_poss.back().pos)) {
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
    for (const valid_pos_t pos_stat : valid_poss) {
        const uint32_t pos = pos_stat.pos;
        const vc_variants1_val_t &var = ht[pos];
        const uint8_t var_stat = pos_stat.stat;
        const uint8_t var_type = var.type;
        if (var_stat != FLAG_VARSTAT_ACCEPTED && var_stat != FLAG_VARSTAT_UNSURE) {
            continue;
        }

        ck.varcalls.push_back(ta_t{});
        ck.varcalls.back().pos = pos;
        ck.varcalls.back().ref_depth = get_ref_depth(var);
        set_variant_depth(ck.varcalls.back(), variant_get_all_allele_coverage(var).cov.cov_tot_all);

        if (var_stat == FLAG_VARSTAT_ACCEPTED) {
            ck.varcalls.back().is_used = TA_STAT_ACCEPTED;
        } else {
            ck.varcalls.back().is_used = TA_STAT_UNSURE;
        }

        if (var_stat == FLAG_VARSTAT_UNSURE) {  // save all alleles
            for (const vc_allele_t &candidate : var.alleles) {
                ck.varcalls.back().alleles.push_back(candidate.allele);
                ck.varcalls.back().allele_depths.push_back(candidate.cov.cov_tot_all);
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
                    assert(var.alleles[0].allele[0] != SENTINEL_REF_ALLELE_INT);
                    ck.varcalls.back().alleles.push_back(var.alleles[0].allele);
                    ck.varcalls.back().allele_depths.push_back(var.alleles[0].cov.cov_tot_all);
                    ck.varcalls.back().type = TA_TYPE_HOM;
                } else {
                    assert(var_type == FLAG_VAR_HET);
                    ck.varcalls.back().alleles.push_back(var.alleles[0].allele);
                    ck.varcalls.back().alleles.push_back(var.alleles[1].allele);
                    ck.varcalls.back().allele_depths.push_back(var.alleles[0].cov.cov_tot_all);
                    ck.varcalls.back().allele_depths.push_back(var.alleles[1].cov.cov_tot_all);
                    ck.varcalls.back().type = TA_TYPE_HET;
                }
            } else {
                assert(var.alleles[0].allele[0] != SENTINEL_REF_ALLELE_INT);
                assert(var.alleles[1].allele[0] != SENTINEL_REF_ALLELE_INT);
                ck.varcalls.back().alleles.push_back(var.alleles[0].allele);
                ck.varcalls.back().alleles.push_back(var.alleles[1].allele);
                ck.varcalls.back().allele_depths.push_back(var.alleles[0].cov.cov_tot_all);
                ck.varcalls.back().allele_depths.push_back(var.alleles[1].cov.cov_tot_all);
                ck.varcalls.back().type = TA_TYPE_HETMULTI;
            }
        }
    }

    // store: variants on reads and reads of a given variants
    std::vector<qa_t> new_;
    for (uint32_t i_read = 0; i_read < ck.reads.size(); i_read++) {
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
                } else if (ta.alleles.size() > 1 && (ta.alleles[1] == allele)) {
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

std::string variant_fullinfo_to_string(const variant_fullinfo_t &v) {
    return fmt::format(
            "valid={} conf={} phased={},{} pos={},{} qual={},{} ref={},{} alt={},{} geno={},{} "
            "ref_depth={} alt_depths={},{} min_depth={} median_depth={}",
            v.is_valid ? "true" : "false", v.is_confident ? "true" : "false",
            v.is_phased0 ? "true" : "false", v.is_phased1 ? "true" : "false", v.pos0, v.pos1,
            v.qual0, v.qual1, v.ref_allele_seq0, v.ref_allele_seq1, v.alt_allele_seq0,
            v.alt_allele_seq1, std::string(v.genotype0, 3), std::string(v.genotype1, 3),
            v.ref_depth, v.alt_depth0, v.alt_depth1, v.min_depth, v.median_depth);
}

bool operator==(const variant_fullinfo_t &a, const variant_fullinfo_t &b) {
    return std::tie(a.is_confident, a.is_multi_allele, a.ref_depth, a.min_depth, a.median_depth,
                    a.pos0, a.qual0, a.ref_allele_seq0, a.alt_allele_seq0, a.is_phased0,
                    a.genotype0[0], a.genotype0[1], a.genotype0[2], a.alt_depth0, a.pos1, a.qual1,
                    a.ref_allele_seq1, a.alt_allele_seq1, a.is_phased1, a.genotype1[0],
                    a.genotype1[1], a.genotype1[2], a.alt_depth1) ==
           std::tie(b.is_confident, b.is_multi_allele, b.ref_depth, b.min_depth, b.median_depth,
                    b.pos0, b.qual0, b.ref_allele_seq0, b.alt_allele_seq0, b.is_phased0,
                    b.genotype0[0], b.genotype0[1], b.genotype0[2], b.alt_depth0, b.pos1, b.qual1,
                    b.ref_allele_seq1, b.alt_allele_seq1, b.is_phased1, b.genotype1[0],
                    b.genotype1[1], b.genotype1[2], b.alt_depth1);
};

phase_return_t kadayashi_local_haptagging_simple_single_region(samFile *fp_bam,
                                                               hts_idx_t *fp_bai,
                                                               sam_hdr_t *fp_header,
                                                               const faidx_t *fai,
                                                               const std::string_view ref_name,
                                                               const uint32_t ref_start,
                                                               const uint32_t ref_end,
                                                               const pileup_pars_t &pp) {
    phase_return_t ret;

    dorado::secondary::BamFileView hf{.fp = fp_bam, .idx = fp_bai, .hdr = fp_header};

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
            const read_t &read = ck.reads[readID];
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

std::unordered_map<std::string, int> kadayashi_simple_single_region_wrapper(
        samFile *fp_bam,
        hts_idx_t *fp_bai,
        sam_hdr_t *fp_header,
        const faidx_t *fai,
        const std::string_view ref_name,
        const uint32_t ref_start,
        const uint32_t ref_end,
        const std::string_view readgroup,
        const bool disable_interval_expansion,
        const int min_base_quality,
        const int min_varcall_coverage,
        const float min_varcall_fraction,
        const int max_clipping,
        const int min_strand_cov,
        const float min_strand_cov_frac,
        const float max_gapcompressed_seqdiv) {
    auto result = kadayashi_simple_single_region_wrapper1(
            fp_bam, fp_bai, fp_header, fai, ref_name, ref_start, ref_end, readgroup,
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
                                                    const std::string_view readgroup,
                                                    const bool disable_interval_expansion,
                                                    const int min_base_quality,
                                                    const int min_varcall_coverage,
                                                    const float min_varcall_fraction,
                                                    const int max_clipping,
                                                    const int min_strand_cov,
                                                    const float min_strand_cov_frac,
                                                    const float max_gapcompressed_seqdiv,
                                                    const bool ambig_ref) {
    dorado::secondary::BamFileView hf_view{fp_bam, fp_bai, fp_header};

    phase_return_t phasing_result;
    phasing_result = kadayashi_simple_single_region_wrapper1(
            fp_bam, fp_bai, fp_header, fai, ref_name, ref_start, ref_end, readgroup,
            disable_interval_expansion, min_base_quality, min_varcall_coverage,
            min_varcall_fraction, max_clipping, min_strand_cov, min_strand_cov_frac,
            max_gapcompressed_seqdiv);

    const pileup_pars_t pp_phased_round{.readgroup = std::string(readgroup),
                                        .min_base_quality = min_base_quality,
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

    const std::string span_s = create_region_string(ref_name, ref_start, ref_end);
    const std::string refseq_s = kadayashi::hts_utils::fetch_seq(fai, span_s);

    // variant phasing+genotype
    ck_derive_variant_genophase_from_phased_read(ck);

    // consolidate variants locally
    ck_absorb_variants_covered_by_deletion_run_on_other_hap(ck, phasing_result.phasing_breakpoints,
                                                            refseq_s);

    // produce variants (no phaseblock info)
    ck_and_varcall_result_t ret;
    varcall_result_internal_t &vr = ret.vr;
    for (const ta_t &varcall : ck.varcalls) {
        if (varcall.is_used == TA_STAT_UNCALLED) {
            continue;
        }
        const variant_fullinfo_t var =
                derive_variant_fullinfo_from_varcall(varcall, refseq_s, ref_start, ambig_ref);
        if (var.is_valid) {
            vr.variants.push_back(var);
        }
    }

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
                                                     const std::string_view readgroup,
                                                     const bool disable_interval_expansion,
                                                     const int min_base_quality,
                                                     const int min_varcall_coverage,
                                                     const float min_varcall_fraction,
                                                     const int max_clipping,
                                                     const int min_strand_cov,
                                                     const float min_strand_cov_frac,
                                                     const float max_gapcompressed_seqdiv,
                                                     const bool ambig_ref) {
    ck_and_varcall_result_t ck_and_vr = kadayashi_phase_and_varcall(
            fp_bam, fp_bai, fp_header, fai, ref_name, ref_start, ref_end, readgroup,
            disable_interval_expansion, min_base_quality, min_varcall_coverage,
            min_varcall_fraction, max_clipping, min_strand_cov, min_strand_cov_frac,
            max_gapcompressed_seqdiv, ambig_ref);
    chunk_t &ck = ck_and_vr.ck;

    if (!ck.is_valid) {
        return {.qname2hp = std::move(ck_and_vr.vr.qname2hp), .variants = {}};
    }

    std::vector<std::pair<uint32_t, uint32_t>> read_endpoints;
    read_endpoints.reserve(std::size(ck.reads));
    for (const auto &read : ck.reads) {
        // limit read spans to the phasing interval so we don't double count reads spanning
        // the ends of the interval
        read_endpoints.emplace_back(std::make_pair(std::max(read.start_pos, ck.abs_start),
                                                   std::min(read.end_pos, ck.abs_end)));
    }
    std::sort(read_endpoints.begin(), read_endpoints.end());

    varcall_result_t ret{.qname2hp = ck_and_vr.vr.qname2hp,
                         .variants = {},
                         .phasing_breakpoints = ck_and_vr.vr.phasing_breakpoints,
                         .read_endpoints = read_endpoints};

    if (!ck.varcalls.empty()) {
        ret.variants.reserve(std::size(ck_and_vr.vr.variants));
        for (const variant_fullinfo_t &var : ck_and_vr.vr.variants) {  // return VCF-ready variants
            if (var.is_valid) {
                dorado::secondary::Variant v = convert_fullinfo_var_to_dorado_style(var, false);
                if (dorado::secondary::is_valid(v)) {
                    ret.variants.emplace_back(std::move(v));
                }
            }
        }
    }

    return ret;
}

struct medaka_feature_matrix_features_t {
    std::string qn;
    uint16_t flag;
    uint8_t mapq;
};

struct medaka_feature_matrix_expansion_entry_t {
    uint32_t pos;
    uint32_t len;
};

struct medaka_feature_matrix_lane_tracker_t {
    int32_t laneID;
    uint32_t last_pos;
};

struct medaka_feature_matrix_lane_tracker_compare_t {
    bool operator()(const medaka_feature_matrix_lane_tracker_t &a,
                    const medaka_feature_matrix_lane_tracker_t &b) const {
        return a.last_pos > b.last_pos;
    }
};

static medaka_feature_matrix_features_t gen_medaka_feature_matrix_parse_features_aln_record(
        dorado::BamPtr &aln) {
    return medaka_feature_matrix_features_t{
            .qn = bam_get_qname(aln.get()), .flag = aln.get()->core.flag, .mapq = aln->core.qual};
}

static inline bool sort_qa_t_ins_last(const qa_t &a, const qa_t &b) {
    assert(!a.allele.empty());
    assert(!b.allele.empty());
    if (a.pos != b.pos) {
        return a.pos < b.pos;
    } else {  // DEL<SNP<INS
        if (a.allele.back() == VAR_OP_I) {
            return false;
        } else if (b.allele.back() == VAR_OP_I) {
            return true;
        } else {
            return a.allele.back() < b.allele.back();
        }
    }
}

static size_t get_qa_t_allele_len(const qa_t &var) {
    if (var.allele.size() < 2) {
        throw std::runtime_error{std::string("[kdys::") + __func__ + "] invalid allele size"};
    }
    return var.allele.size() - 1;
}

static int8_t get_snp_qv_medaka_style(const read_t &read) {
    int n_mismatches = 0;
    int n_ins = 0;
    for (const qa_t &var : read.vars) {
        if (var.allele.back() == VAR_OP_X) {
            n_mismatches++;
        } else if (var.allele.back() == VAR_OP_I) {
            n_ins += get_qa_t_allele_len(var);
        }
    }

    // adapted from dorado util's `compute_accuracy_from_cigar`
    const int n_MX = read.end_pos <= read.start_pos ? 0 : read.end_pos - read.start_pos + n_ins;
    const double acc_x =
            n_MX == 0 ? 0.0
                      : std::clamp((1.0 - (static_cast<double>(n_mismatches)) / n_MX), 0.0, 1.0);
    const double err_prob = 1.0 - acc_x;
    const double quality_score = dorado::utils::compute_quality_score(err_prob);

    return static_cast<int8_t>(nearbyint(quality_score));
}

static double get_snp_accuracy_dorado_style(const read_t &read, const char *qn) {
    // ref: dorado::utils::compute_accuracy_from_cigar
    if (read.end_pos <= read.start_pos) {
        return 0.0;
    }
    if (read.vars.empty()) {
        return 1.0;
    }
    int n_mismatches = 0;
    int n_ins = 0;
    int n_del = 0;
    for (const qa_t &var : read.vars) {
        const uint8_t cigar_op = var.allele.back();
        const int cigar_size = static_cast<int>(get_qa_t_allele_len(var));
        if (cigar_op == VAR_OP_I) {
            n_ins += cigar_size;
        } else if (cigar_op == VAR_OP_D) {
            n_del += cigar_size;
        } else if (cigar_op == VAR_OP_X) {
            n_mismatches++;
        }
    }
    const int span = static_cast<int>(read.end_pos - read.start_pos);
    if (span <= n_del) {
        throw std::runtime_error{std::string("[kdys::") + __func__ +
                                 "] read has abnormal deletion length? qn " + qn + "aln start " +
                                 std::to_string(read.start_pos) + " end " +
                                 std::to_string(read.end_pos) + " del " + std::to_string(n_del)};
    }
    const int tot = span - n_del + n_ins;
    return 1.0 - (static_cast<double>(n_mismatches) / tot);
}

struct featmatgen_chunk_t {
    // parameters
    const std::string_view refname;
    const uint32_t itvl_start;
    const uint32_t itvl_end;
    const MedakaFeatureMatrixOptions &options;
    const int n_features;

    // helper stats
    uint32_t n_downsample_filtered;
    uint32_t n_md_tag_filtered;

    // helper for collecting refseq from read pileup
    std::vector<uint8_t> refseq_substring;
    std::vector<uint8_t> read_seqi{};  // temp buffer to be used by bam_seqi

    // helper declaration: depending on the flags, we might load haptag
    // from bam rather than the ht from input.
    str2int_t qname2hp_bam{};

    // storage: reads and their auxiliary data
    std::vector<read_t> all_reads{};
    std::vector<int8_t> all_mapq{};    // can be put into read_t
    std::vector<int8_t> all_snp_qv{};  // can be put into read_t
    std::vector<int8_t> all_dtype{};
    std::unordered_map<uint32_t, uint32_t> all_ins{};
    std::unordered_map<uint32_t, std::string> readID2qn{};
    std::vector<std::vector<uint8_t>> all_reads_quals{};
    std::vector<std::vector<int8_t>> all_dwells{};
};
static void gen_medaka_feature_matrix_store_reads_from_bam(dorado::secondary::BamFileView &hf,
                                                           featmatgen_chunk_t &gck) {
    const uint8_t seqi2int[16] = {0,        1 /*A1*/, 2 /*C2*/, 0, 3 /*G4*/, 0, 0, 0,
                                  4 /*T8*/, 0,        0,        0, 0,        0, 0, 5 /*N15*/};

    // input files
    const std::string itvl = create_region_string(gck.refname, gck.itvl_start, gck.itvl_end);
    dorado::secondary::HtsItrPtr bamitr(sam_itr_querys(hf.idx, hf.hdr, itvl.c_str()));
    dorado::BamPtr aln(bam_init1());

    // helpers for read downsampling (adapted from pileup_ht)
    constexpr bool ENABLE_DOWNSAMPLE = true;
    gck.n_downsample_filtered = 0;
    std::vector<int> downsample_counter;
    const int n_counter = (gck.itvl_end - gck.itvl_start) / DOWNSAMPLE_WINDOW + 1;
    if (ENABLE_DOWNSAMPLE) {
        downsample_counter.resize(n_counter, 0);
    }

    uint32_t n_reads = 0;
    gck.n_md_tag_filtered = 0;
    std::unordered_set<std::string> seen_qnames;
    while (sam_itr_next(hf.fp, bamitr.get(), aln.get()) >= 0) {
        const char *qn = bam_get_qname(aln.get());
        const uint32_t r_start_pos = static_cast<uint32_t>(aln.get()->core.pos);
        const uint32_t r_end_pos = static_cast<uint32_t>(bam_endpos(aln.get()));

        const bool md_is_ok = sancheck_MD_tag_exists_and_is_valid(aln.get());
        if (!md_is_ok) {  // TODO: remove the requirement on MD tags
            ++gck.n_md_tag_filtered;
            continue;
        }

        if (to_exclude_by_flags(aln.get(), 4 | 256 | 2048)) {
            continue;
        }
        if (to_exclude_by_low_mapq(aln.get(), gck.options.min_mapq)) {
            continue;
        }

        // filter by readgroup (adapted from medaka_bamiter.cpp)
        if (!gck.options.readgroup.empty()) {
            const uint8_t *rg = bam_aux_get(aln.get(), "RG");
            if (rg) {
                errno = 0;  // reset
                const char *rg_val = bam_aux2Z(rg);
                if (errno == EINVAL) {
                    continue;
                }
                if (gck.options.readgroup != rg_val) {
                    continue;
                }
            }
        }

        // filter by one specified tag (adapted from medaka_bamiter.cpp)
        if (!gck.options.tag_name.empty()) {
            const uint8_t *tag = bam_aux_get(aln.get(), gck.options.tag_name.c_str());
            if (tag == nullptr) {
                if (gck.options.tag_keep_missing) {
                    break;
                } else {
                    continue;
                }
            }
            const int32_t tag_value = static_cast<int32_t>(bam_aux2i(tag));
            if (errno == EINVAL) {
                continue;
            }
            if (tag_value != gck.options.tag_value) {
                continue;
            }
        }

        // Consider depth-based filtering.
        // Don't update the counter yet, we might want to ignore the read
        // based on other criteria.
        if (ENABLE_DOWNSAMPLE) {
            const bool read_may_be_accepted = read_downsampling_query_or_update_counter(
                    read_downsampling_rw::READ_DOWNSAMPLING_QUERY_ONLY, r_start_pos, r_end_pos,
                    gck.itvl_start, gck.itvl_end, downsample_counter, DOWNSAMPLE_WINDOW,
                    DOWNSAMPLE_READCAP);
            if (!read_may_be_accepted) {
                gck.n_downsample_filtered++;
                continue;  // go parse the next read
            }
        }

        // Parse the alignment record to collect variants of the read;
        // consider to filter based on mismatch seq identity.
        medaka_feature_matrix_features_t features =
                gen_medaka_feature_matrix_parse_features_aln_record(aln);
        read_t r{
                .start_pos = static_cast<uint32_t>(aln.get()->core.pos),
                .end_pos = static_cast<uint32_t>(bam_endpos(aln.get())),
                .ID = n_reads,
                .vars = {},
                .hp = HAPTAG_UNPHASED,
                .votes_diploid = {0, 0},
                .strand = !!(features.flag & 16),
                .de = 0.0f,
                .left_clip_len = 0,
                .right_clip_len = 0,
        };
        const bool parse_ok = parse_variants_for_one_read(
                aln.get(), r.vars, 0 /*pp.min_base_quality*/, &r.left_clip_len, &r.right_clip_len,
                false /*retain_SNP_only*/, nullptr);
        const double seqdiv_mismatch_only = get_snp_accuracy_dorado_style(r, qn);
        if ((seqdiv_mismatch_only < gck.options.min_snp_accuracy) || !parse_ok) {
            continue;
        }
        if (!seen_qnames.emplace(qn).second) {
            continue;
        }

        // Read is accepted. Update the downsampling counter.
        read_downsampling_query_or_update_counter(
                read_downsampling_rw::READ_DOWNSAMPLING_QUERY_AND_UPDATE, r_start_pos, r_end_pos,
                gck.itvl_start, gck.itvl_end, downsample_counter, DOWNSAMPLE_WINDOW,
                DOWNSAMPLE_READCAP);

        // save haptag from the alignment record if requested.
        if (gck.options.include_haplotype_column &&
            ((gck.options.hap_source == haplotagsource::USE_BAM_HAP_TAG))) {
            const uint8_t *const tag = bam_aux_get(aln.get(), "HP");
            if (tag) {
                gck.qname2hp_bam[qn] = static_cast<int>(bam_aux2i(tag));
            }
        }

        // calculate snp qv now, because we will adjust start_pos and
        // end_pos of read later.
        gck.all_snp_qv.push_back(get_snp_qv_medaka_style(r));

        // adjust deletion position:
        // `parse_variants_for_one_read` saves deletion positions one base
        // prior to their true positions due to vcf formating needs.
        for (qa_t &var : r.vars) {
            if (var.allele.back() == VAR_OP_D) {
                var.pos++;
            }
        }

        // (workaround for lack of refseq)
        const uint8_t *seqdata = bam_get_seq(aln.get());
        gck.read_seqi.clear();
        const int seq_len = aln.get()->core.l_qseq;
        for (int i = r.left_clip_len; i < seq_len; i++) {
            gck.read_seqi.push_back(seqi2int[bam_seqi(seqdata, i)]);  // unset0; ACGTN 12345
        }

        // Go through the read variants,
        // collect insertions, also drop the variants outside of
        // the query region (adjust read start and end position accordingly).
        uint32_t readadj_offset_from_ins = 0;  // Accumulate the #bases from insertions on the read
                                               // to the left of the query region.
        uint32_t readadj_offset_from_del = 0;  // (Similar to above but for deletions.)
        uint32_t readadj_first_del_shift = 0;  // If read aln start is to the left of query, and
                                               // a deletion overlaps with that left edge, we need
                                               // to record the length of [left edge, end of del)
                                               // in order to adjust the read's start position
                                               // correctly.
        if (!r.vars.empty()) {
            int32_t first_viable_i = -1;
            int32_t last_viable_i = -1;
            std::sort(r.vars.begin(), r.vars.end(), sort_qa_t_ins_last);

            for (int i = 0; i < static_cast<int32_t>(r.vars.size()); i++) {
                const qa_t &var = r.vars[i];
                const uint32_t var_len = get_qa_t_allele_len(var);
                const uint8_t cigar = var.allele.back();
                if (var.pos < gck.itvl_start) {
                    if (cigar == VAR_OP_I) {
                        readadj_offset_from_ins += var_len;
                    } else if (cigar == VAR_OP_D) {
                        if (var.pos + var_len >= gck.itvl_start) {
                            readadj_first_del_shift = var.pos + var_len - gck.itvl_start;
                            readadj_offset_from_del += var_len;
                        } else {
                            readadj_offset_from_del += var_len;
                        }
                    }
                    continue;
                }
                if (var.pos >= gck.itvl_end) {
                    break;
                }

                if (first_viable_i < 0) {
                    first_viable_i = i;
                }
                last_viable_i = i;

                if (var_len >= MEDAKA_FEATURE_MATRIX_MAX_VAR_LEN) {
                    spdlog::error("[kdys::{}] variant too long ({}-1) read qn {} var pos {}",
                                  __func__, var.allele.size(), features.qn, var.pos);
                    break;
                }

                if (cigar == VAR_OP_I) {
                    auto [it, inserted] = gck.all_ins.emplace(var.pos, var_len);
                    if (!inserted && var_len > it->second) {
                        it->second = var_len;
                    }
                }
            }

            // (remove variants whose start position is
            // outside of the query region.)
            if (!r.vars.empty()) {
                if (first_viable_i < 0 || last_viable_i < 0) {
                    r.vars.clear();
                } else {
                    if (first_viable_i < 0 || last_viable_i < 0) {
                        throw std::runtime_error{std::string("[kdys::") + __func__ +
                                                 "] removing oob variants from read failed: qn " +
                                                 qn + "interval " + itvl};
                    } else {
                        const uint32_t right = r.vars.size() - last_viable_i + 1;
                        if (first_viable_i > 0) {
                            r.vars.erase(r.vars.begin(), r.vars.begin() + first_viable_i);
                        }
                        if (right > 0) {
                            r.vars.erase(r.vars.end() - right + 2, r.vars.end());
                        }
                    }
                }
            }
        }  // (drop read variants that are outside of the query region)

        // consider adjust read start and end position (read vars were truncated like so above)
        const uint32_t r_raw_start_pos = r.start_pos;
        r.start_pos = gck.itvl_start > r.start_pos ? gck.itvl_start + readadj_first_del_shift
                                                   : r.start_pos;
        r.end_pos = std::min<uint32_t>(r.end_pos, gck.itvl_end);

        // (workaround for refseq: fill the substring refseq)
        // (we also need the `offset_on_read` to slice base qualities and dwells, though we don't
        //  need the block figuring out where to skip the variants - only bases that aren't val5 and val0
        //  in the expanded read will have base quality anyways. As a result, this way we
        //   do not get the the exact on-read end position of them, which is fine.)
        uint32_t end_pos = r.vars.empty() ? r.end_pos : r.vars[0].pos;
        uint32_t offset_on_ref = r.start_pos - gck.itvl_start;
        uint32_t offset_on_read =
                r.start_pos - r_raw_start_pos + readadj_offset_from_ins - readadj_offset_from_del;

        const uint8_t *quals = bam_get_qual(aln.get());
        gck.all_reads_quals.emplace_back(quals + offset_on_read + r.left_clip_len,
                                         quals + aln.get()->core.l_qseq);

        // (calculate dwell and shift it too)
        std::vector<int8_t> dwells;
        dorado::secondary::calculate_dwells(aln.get(), dwells);
        gck.all_dwells.emplace_back(dwells.begin() + offset_on_read + r.left_clip_len,
                                    dwells.end());
        // (first)
        for (uint32_t i = r.start_pos; i < end_pos; i++) {
            if (gck.refseq_substring[offset_on_ref] != 0 &&
                gck.refseq_substring[offset_on_ref] != gck.read_seqi[offset_on_read]) {
                spdlog::debug(
                        "[wut][r {} {}] overwriting refseq substring ({}->{}) shouldn't "
                        "happen#1 ; pos is {}\n",
                        n_reads, qn, gck.refseq_substring[offset_on_ref],
                        gck.read_seqi[offset_on_read], offset_on_ref + gck.itvl_start);
            }
            gck.refseq_substring[offset_on_ref] = gck.read_seqi[offset_on_read];
            offset_on_ref++;
            offset_on_read++;
        }

        // (rest: var-matches-var-matches...end)
        for (uint32_t i = 0; i < r.vars.size(); i++) {
            const qa_t &var = r.vars[i];
            const uint32_t var_len = get_qa_t_allele_len(var);
            const uint8_t cigar = var.allele.back();

            end_pos = (i + 1 >= r.vars.size()) ? r.end_pos : r.vars[i + 1].pos;
            if (cigar == VAR_OP_D) {
                offset_on_ref += var_len;
            } else if (cigar == VAR_OP_X) {
                offset_on_ref++;
                offset_on_read++;
            } else if (cigar == VAR_OP_I) {
                bool skip_ref_base_before_ins = false;
                if (i > 0) {
                    const qa_t &prev_var = r.vars[i - 1];
                    const uint8_t prev_cigar = prev_var.allele.back();
                    const size_t prev_len = get_qa_t_allele_len(prev_var);
                    const uint32_t prev_pos = prev_var.pos;
                    if ((prev_cigar == VAR_OP_X) && (prev_pos == var.pos)) {
                        skip_ref_base_before_ins = true;
                    } else if ((prev_cigar == VAR_OP_D) && (prev_pos + prev_len > var.pos)) {
                        skip_ref_base_before_ins = true;
                    }
                }
                if (!skip_ref_base_before_ins) {
                    gck.refseq_substring[offset_on_ref] = gck.read_seqi[offset_on_read];
                }
                offset_on_ref++;
                offset_on_read++;
                offset_on_read += var_len;
            } else {
                spdlog::error("[kdys::{}] r {} saw invalid cigar {}\n", __func__, qn, cigar);
            }
            for (uint32_t j = offset_on_ref + gck.itvl_start; j < end_pos; j++) {
                if (gck.refseq_substring[offset_on_ref] != 0 &&
                    gck.refseq_substring[offset_on_ref] != gck.read_seqi[offset_on_read]) {
                    spdlog::error(
                            "[kdys::{}] r {} overwriting refseq substring ({}->{}) "
                            "shouldn't happen#2; pos is {}\n",
                            __func__, qn, gck.refseq_substring[offset_on_ref],
                            gck.read_seqi[offset_on_read], offset_on_ref + gck.itvl_start);
                }
                gck.refseq_substring[offset_on_ref] = gck.read_seqi[offset_on_read];
                offset_on_ref++;
                offset_on_read++;
            }
        }
        gck.readID2qn[n_reads] = bam_get_qname(aln.get());
        gck.all_mapq.push_back(static_cast<int8_t>(aln.get()->core.qual));
        gck.all_reads.emplace_back(std::move(r));

        // adapted from medaka
        int32_t dtype = 0;
        if (gck.options.num_dtypes > 1) {
            bool failed = false;
            if (gck.options.num_dtypes > 1) {
                char *tag_val = nullptr;
                const uint8_t *tag = bam_aux_get(aln.get(), "DT");
                if (tag == NULL) {  // tag isn't present
                    failed = true;
                } else {
                    tag_val = bam_aux2Z(tag);
                    failed = errno == EINVAL;
                }
                if (!failed) {
                    bool found = false;
                    for (dtype = 0; dtype < gck.options.num_dtypes; ++dtype) {
                        if (tag_val && ((gck.options.dtypes[dtype] == tag_val))) {
                            found = true;
                            break;
                        }
                    }
                    failed = !found;
                }
                if (failed) {
                    spdlog::error("[kdys::{}] Datatype not found for read {}", __func__, qn);
                }
            }
        }
        gck.all_dtype.push_back(static_cast<int8_t>(dtype));

        n_reads++;
    }  // iterate through all reads
    if (gck.n_md_tag_filtered > 0) {
        LOG_DEBUG("[kdys::{}] skipped {} reads without valid MD tags", __func__,
                  gck.n_md_tag_filtered);
    }
}

static void gen_medaka_feature_matrix_insert_to_matrix(
        MedakaFeatureMatrix &mfm,
        const std::unordered_map<std::string, int32_t> &qname2hp,
        const featmatgen_chunk_t &gck,
        uint32_t i_read,
        int laneID,
        uint32_t read_offset_start,
        uint32_t read_offset,
        const std::vector<int8_t> &expanded_read,
        const std::vector<int8_t> &expanded_read_quals,
        const std::vector<int8_t> &expanded_read_dwells) {
    std::vector<int8_t> &matrix = mfm.matrix;
    std::vector<int64_t> &majors = mfm.major;
    std::vector<int64_t> &minors = mfm.minor;

    // prep other scalar fields
    const read_t read = gck.all_reads[i_read];
    const uint8_t strand = read.strand;
    const int8_t mapq = gck.all_mapq[i_read];
    const int8_t snp_qv = gck.all_snp_qv[i_read];
    const int8_t dtype = gck.all_dtype[i_read];
    uint8_t haptag = HAPTAG_UNPHASED;
    if (gck.options.include_haplotype_column) {
        switch (gck.options.hap_source) {
        case haplotagsource::FORCE_UNPHASED:
            break;
        case haplotagsource::USE_BAM_HAP_TAG: {
            const auto it = gck.readID2qn.find(i_read);
            assert(it != gck.readID2qn.cend());
            const auto it_hp = gck.qname2hp_bam.find(it->second);
            if (it_hp != gck.qname2hp_bam.cend()) {
                haptag = static_cast<uint8_t>(it_hp->second);
            }
        } break;
        case haplotagsource::USE_TAG_FROM_HASHTABLE: {
            const auto it = gck.readID2qn.find(i_read);
            assert(it != gck.readID2qn.cend());
            const auto it_hp = qname2hp.find(it->second);
            if (it_hp != qname2hp.cend()) {
                haptag = static_cast<uint8_t>(it_hp->second);
            }
        } break;
        }
    }

    // now migrate to the matrix
    const uint32_t i_pos_left = read_offset_start;
    const uint32_t i_pos_right = std::min<uint32_t>(read_offset, expanded_read.size());
    const uint32_t shift_step = mfm.buffer_reads * mfm.featlen;
    uint32_t shift_base = shift_step * i_pos_left + mfm.featlen * laneID;
    uint32_t ref_pos = majors[i_pos_left] - gck.itvl_start;
    for (uint32_t i = i_pos_left; i < i_pos_right; i++) {
        uint32_t shift = shift_base;

        // the base
        if (expanded_read[i] == 6) {
            if (minors[i] != 0) {
                spdlog::error(
                        "[kdys::{}] impossible: match is minor pos {} (minor={}) "
                        "{}:{}-{} . Matrix will be incorrect .\n",
                        __func__, majors[i], minors[i], gck.refname, gck.itvl_start, gck.itvl_end);
            }
            matrix[shift] = gck.refseq_substring[ref_pos];
            shift++;
            ref_pos++;
        } else {
            matrix[shift] = expanded_read[i];
            shift++;
            if ((minors[i] == 0) && (expanded_read[i] == 5)) {  // deletion
                ref_pos++;
            } else if ((expanded_read[i] != 0) && (expanded_read[i] != 5) &&
                       (minors[i] == 0)) {  // mismatch
                ref_pos++;
            }
        }

        // other entires
        matrix[shift] = expanded_read_quals[i];
        shift++;

        matrix[shift] = strand == 0 ? 1 : -1;  // medaka convention
        shift++;

        matrix[shift] = mapq;
        shift++;

        if (gck.options.include_dwells) {
            matrix[shift] = expanded_read_dwells[i];
            shift++;
        }
        if (gck.options.include_haplotype_column) {
            matrix[shift] = (haptag == HAPTAG_UNPHASED ? 0 : haptag);  // medaka convention
            shift++;
        }
        if (gck.options.include_snp_qv) {
            matrix[shift] = snp_qv;
            shift++;
        }
        if (gck.options.num_dtypes > 1) {
            matrix[shift] = dtype;
            shift++;
        }

        shift_base += shift_step;
    }
}

MedakaFeatureMatrix gen_medaka_feature_matrix(
        dorado::secondary::BamFileView &hf,
        std::string_view refname,
        const uint32_t itvl_start,
        const uint32_t itvl_end,
        const std::unordered_map<std::string, int32_t> &qname2hp,
        const MedakaFeatureMatrixOptions &options) {
    // This function goes through the bam region once to collect the variants of the reads.
    // After that we pack the reads into lanes (like pseudo reads), where each lane contain
    // at least one non-overlapping reads. Since all variants in the query intervals
    // are known, we know the lengths and the locations of minor columns.
    // Feature matrix is then allocated.
    // The feature matrix is in shape [n_pos, n_reads, n_features] and we stored
    // the alignment as individual aligned reads, thus there will an implicit transposition.
    // We iterate through reads. For each read we first store its info into
    // a 1D array `expanded_read`, which in then inserted into the matrix in the transposed order.

    constexpr bool WUT_VERBOSE = false;
    if (itvl_end <= itvl_start) {
        spdlog::error("[kdys::{}] invalid query region: ref {} start {} end {}. Skipping.",
                      __func__, refname, itvl_start, itvl_end);
        MedakaFeatureMatrix ret(0, 0, 0, 0, 1, 0);
        return ret;
    }

    featmatgen_chunk_t gck = {
            // pars
            .refname = refname,
            .itvl_start = itvl_start,
            .itvl_end = itvl_end,
            .options = options,
            .n_features = 4 + (options.include_dwells ? 1 : 0) +
                          (options.include_haplotype_column ? 1 : 0) +
                          (options.include_snp_qv ? 1 : 0) + (options.num_dtypes > 1),
            // helper: stats
            .n_downsample_filtered = 0,
            .n_md_tag_filtered = 0,
            // helper: refseq
            .refseq_substring = std::vector<uint8_t>(itvl_end - itvl_start, 0),
            .read_seqi = {},
            // helper: optional holder of haptags
            .qname2hp_bam = {},
            // read variant storages
            .all_reads = {},
            .all_mapq = {},
            .all_snp_qv = {},
            .all_dtype = {},
            .all_ins = {},
            .readID2qn = {},
            .all_reads_quals = {},
            .all_dwells = {}};
    gen_medaka_feature_matrix_store_reads_from_bam(hf, gck);

    if (gck.all_reads.empty()) {
        LOG_DEBUG("[kdys::{}] requested interval {}:{}-{} inserted no reads", __func__, refname,
                  itvl_start, itvl_end);
        MedakaFeatureMatrix ret(0, 0, 0, 0, 1, 0);
        return ret;
    } else {
        if constexpr (DEBUG_LOCAL_HAPLOTAGGING) {
            spdlog::debug(
                    "[kdys::{}] requested interval {}:{}-{} inserted {} reads (downsample filtered "
                    "{})",
                    __func__, refname, itvl_start, itvl_end, gck.all_reads.size(),
                    gck.n_downsample_filtered);
        }
    }

    // figure out what columns need expansion, and the total size of column expansion
    std::vector<medaka_feature_matrix_expansion_entry_t> expansions;
    uint32_t tot_expansion = 0;
    for (const auto &[pos, var_len] : gck.all_ins) {
        expansions.push_back({.pos = pos, .len = var_len});
        tot_expansion += var_len;
    }
    std::sort(expansions.begin(), expansions.end(),
              [](const medaka_feature_matrix_expansion_entry_t &a,
                 const medaka_feature_matrix_expansion_entry_t &b) { return a.pos < b.pos; });

    // Figure out how to pack the reads.
    // This has to be done before allocating the matrix because we
    // need to know the number of lanes.
    // Reads that overlap with the left or the right boundary of
    // the query interval also need to have their names stored.
    int n_lanes = 0;
    std::vector<int> read2lane(gck.all_reads.size(), -1);
    std::vector<uint32_t> read2matrix_start_pos(gck.all_reads.size(), 0);
    std::vector<std::string> left_qnames;
    std::vector<std::string> right_qnames;
    {
        const int lane_cap = std::min<int>(100, options.max_reads);
        int laneID = 0;
        std::vector<medaka_feature_matrix_lane_tracker_t> lane_lookup;
        std::unordered_map<uint32_t, std::string> tmp_lane2rightqn;

        for (size_t i = 0; i < gck.all_reads.size(); i++) {
            const read_t &read = gck.all_reads[i];

            bool need_new_lane = true;
            int &read_laneID = read2lane[i];
            if (!options.disable_read_packing) {  // then try to place read into an existing lane
                for (medaka_feature_matrix_lane_tracker_t &record : lane_lookup) {
                    if (record.last_pos <= read.start_pos) {
                        need_new_lane = false;
                        read_laneID = record.laneID;
                        read2matrix_start_pos[i] = read.start_pos;
                        record.last_pos = read.end_pos + DORADO_FEATURE_MAT_READ_SENTINAL_LEN;
                        break;
                    }
                }
            }

            if (need_new_lane) {
                if (static_cast<int>(std::size(lane_lookup)) < lane_cap) {
                    read_laneID = laneID++;
                    read2matrix_start_pos[i] = read.start_pos;
                    lane_lookup.push_back(medaka_feature_matrix_lane_tracker_t{
                            .laneID = read_laneID,
                            .last_pos = read.end_pos + DORADO_FEATURE_MAT_READ_SENTINAL_LEN});
                    if (read.start_pos <= itvl_start) {
                        left_qnames.push_back(gck.readID2qn[i]);
                    } else {
                        left_qnames.push_back("");
                    }
                } else if (!options.disable_read_packing) {
                    auto lane_it =
                            std::min_element(std::begin(lane_lookup), std::end(lane_lookup),
                                             [](const medaka_feature_matrix_lane_tracker_t &a,
                                                const medaka_feature_matrix_lane_tracker_t &b) {
                                                 return a.last_pos < b.last_pos;
                                             });
                    if ((lane_it != std::end(lane_lookup)) && (lane_it->last_pos < read.end_pos)) {
                        read_laneID = lane_it->laneID;
                        read2matrix_start_pos[i] = lane_it->last_pos;
                        lane_it->last_pos = read.end_pos + DORADO_FEATURE_MAT_READ_SENTINAL_LEN;
                    } else {
                        continue;
                    }
                } else {
                    continue;
                }
            }

            if (read.end_pos >= itvl_end) {  // `>` is wrong because end_pos was capped at itvl_end.
                tmp_lane2rightqn[read_laneID] = gck.readID2qn[i];
            }
        }
        n_lanes = static_cast<int>(std::size(lane_lookup));

        // (consolidate qnames on the right)
        for (int i_lane = 0; i_lane < n_lanes; i_lane++) {
            right_qnames.push_back(tmp_lane2rightqn[i_lane]);  // the [] here is fine as
                                                               // we do want an empty string when
                                                               // there's no known read that
                                                               // overlaps with the right edge.
        }
        spdlog::trace("[kdys::{}] {}:{}-{} has {} lanes (subject to capping)", __func__, refname,
                      itvl_start, itvl_end, n_lanes);
    }

    // Give placeholder names 1-index to empty entries in left and right qnames
    // because of medaka convention.
    for (uint32_t i = 0, j = 1; i < left_qnames.size(); i++) {
        if (left_qnames[i].empty()) {
            left_qnames[i] = "__blank_" + std::to_string(j++);
        }
    }
    for (uint32_t i = 0, j = 1; i < right_qnames.size(); i++) {
        if (right_qnames[i].empty()) {
            right_qnames[i] = "__blank_" + std::to_string(j++);
        }
    }

    // Allocate the matrix and major & minor positions.
    uint32_t n_pos = tot_expansion + itvl_end - itvl_start;
    MedakaFeatureMatrix ret(n_pos, n_lanes, n_pos, n_lanes, gck.n_features - 4,
                            0);  // medaka impl also hardcodes the last parameter to 0
    ret.read_ids_left = std::move(left_qnames);
    ret.read_ids_right = std::move(right_qnames);
    std::vector<int8_t> &matrix = ret.matrix;
    std::fill(matrix.begin(), matrix.end(), 0);
    std::vector<int64_t> &majors = ret.major;
    std::vector<int64_t> &minors = ret.minor;
    if constexpr (WUT_VERBOSE) {
        spdlog::debug(
                "[wut] alloc: n_pos={} (tot_exp={}) n_reads={} n_lanes(clampped)={} n_feature={}; "
                "mat and "
                "poss sizes: {} {} {}\n",
                n_pos, tot_expansion, gck.all_reads.size(), n_lanes, gck.n_features, matrix.size(),
                majors.size(), minors.size());
    }

    // fill major and minor positions
    {
        uint32_t pos = itvl_start;
        uint32_t i = 0;

        // start ~ first expansion
        {
            uint32_t pos_right = expansions.empty() ? itvl_end : expansions[0].pos;
            uint32_t tmp_pad = expansions.empty() ? 0 : 1;
            while (pos < pos_right + tmp_pad) {
                majors[i] = pos;
                minors[i] = 0;
                i++;
                pos++;
            }
            if (tmp_pad > 0) {
                pos--;
            }
        }

        // from the first expansion
        for (uint32_t j = 0; j < expansions.size(); j++) {
            const medaka_feature_matrix_expansion_entry_t &exp = expansions[j];

            // in the expansion (the minor cols)
            for (uint32_t k = 1; k <= exp.len; k++) {
                minors[i] = k;
                majors[i] = pos;
                i++;
            }
            pos++;

            // after the expansion
            uint32_t pos_right = (j + 1 >= expansions.size()) ? itvl_end : expansions[j + 1].pos;
            uint32_t tmp_pad = (j + 1 >= expansions.size()) ? 0 : 1;
            while (pos < pos_right + tmp_pad) {
                majors[i] = pos;
                minors[i] = 0;
                i++;
                pos++;
            }
            if (tmp_pad > 0) {
                pos--;
            }
        }
    }

    // fill the feature matrix.
    // note:
    //   - when insertion co-exist with other types of variants, insertion goes last.
    //   - within an insertion run, consider right-align the allele if it exists and
    //     right-align was requested. (otherwise it's left align; there is no
    //     "original" alignment configure of an insertion inside the minor columns.)
    // For simplicity, fill a temporary buffer when we expand the read,
    // then insert the buffer into the matrix afterward. Similarly,
    // do this for base qualities and qvs. (note that there are several
    // offset adjusts for the start coordinate on read: left soft clip,
    // my clamping of read start position wrt query interval start,
    // del on read, and ins expansion due to pileup.)
    std::vector<int8_t> expanded_read(ret.buffer_pos, 0);
    std::vector<int8_t> expanded_read_quals(ret.buffer_pos, 0);
    std::vector<int8_t> expanded_read_dwells(ret.buffer_pos, 0);
    for (size_t i_read = 0; i_read < gck.all_reads.size(); i_read++) {
        // Limit the max number of lanes.
        const int laneID = read2lane[i_read];
        if ((laneID < 0) || (laneID >= n_lanes)) {
            continue;
        }

        const read_t &read = gck.all_reads[i_read];

        // count expanded columns prior to the read's start position
        uint32_t offset = read.start_pos - itvl_start;
        for (const medaka_feature_matrix_expansion_entry_t &exp : expansions) {
            if (exp.pos >= read.start_pos) {
                break;
            }
            offset += exp.len;
        }
        if constexpr (WUT_VERBOSE) {
            spdlog::debug("[wut][r {}] start {} , offset {}, offset-start {}\n",
                          gck.readID2qn[i_read], read.start_pos, offset, majors[offset]);
        }
        const uint32_t offset0 = offset;

        // fill expanded_read
        // (from start to the first variant)
        {
            const uint32_t pos_right = read.vars.empty() ? read.end_pos : read.vars[0].pos;
            for (uint32_t pos = read.start_pos; pos < pos_right; pos++) {
                expanded_read[offset] = 6;
                offset++;
                while (offset < minors.size() && minors[offset] != 0) {
                    expanded_read[offset] = 5;
                    offset++;
                }
            }
        }
        // (for each variant and matches after it)
        for (uint32_t i_var = 0; i_var < read.vars.size();) {
            const qa_t &var = read.vars[i_var];
            uint32_t i_var_incre = 0;

            // the variant
            if (var.allele.back() == VAR_OP_X) {
                if constexpr (WUT_VERBOSE) {
                    spdlog::debug("[wut][r {}] pushing X: pos {} offset {} val {}\n",
                                  gck.readID2qn[i_read], var.pos, offset, var.allele[0] + 1);
                }
                expanded_read[offset] = var.allele[0] + 1;
                offset++;
                i_var_incre = 1;

                // if next is col expansion and self doesn't have insertion at
                // the same position, fastforward
                if (i_var + 1 >= read.vars.size()) {
                    while (offset < minors.size() && minors[offset] != 0) {
                        expanded_read[offset] = 5;
                        offset++;
                    }
                } else if (i_var + 1 < read.vars.size() &&
                           !((read.vars[i_var + 1].allele.back() == VAR_OP_I) &&
                             (read.vars[i_var + 1].pos == var.pos))) {
                    while (offset < minors.size() && minors[offset] != 0) {
                        expanded_read[offset] = 5;
                        offset++;
                    }
                }
            } else if (var.allele.back() == VAR_OP_D) {
                if constexpr (WUT_VERBOSE) {
                    spdlog::debug("[wut][r {}] pushing DEL: pos {} offset {} len {}\n",
                                  gck.readID2qn[i_read], var.pos, offset, get_qa_t_allele_len(var));
                }
                for (uint32_t i = 0; i < get_qa_t_allele_len(var); i++) {
                    if (offset >= expanded_read.size()) {
                        break;
                    }
                    expanded_read[offset] = 5;
                    offset++;
                    // if next is col expansion and self doesn't have insertion at
                    // each of the current position, fastforward
                    if (i_var + 1 >= read.vars.size()) {
                        while (offset < minors.size() && minors[offset] != 0) {
                            expanded_read[offset] = 5;
                            offset++;
                        }
                    } else if (i_var + 1 < read.vars.size() &&
                               !((read.vars[i_var + 1].allele.back() == VAR_OP_I) &&
                                 (read.vars[i_var + 1].pos == majors[offset - 1]))) {
                        while (offset < minors.size() && minors[offset] != 0) {
                            expanded_read[offset] = 5;
                            offset++;
                        }
                    }
                }
                i_var_incre = 1;
            } else if (var.allele.back() == VAR_OP_I) {
                // fill in the ref base first , then the right-aligned insertion w/ col exp
                if constexpr (WUT_VERBOSE) {
                    spdlog::debug("[wut][r {}] insertion at {} (offset {} major {}) length {}\n",
                                  gck.readID2qn[i_read].c_str(), var.pos, offset, majors[offset],
                                  get_qa_t_allele_len(var));
                }

                //(ref base; need to check if we have a mismatch or a del previously)
                bool skip_ref_base_before_ins = false;
                uint8_t prev_cigar = 0;
                uint32_t prev_len = 0;
                uint32_t prev_pos = 0;
                if (i_var > 0) {
                    const qa_t &prev_var = read.vars[i_var - 1];
                    prev_cigar = prev_var.allele.back();
                    prev_len = get_qa_t_allele_len(prev_var);
                    prev_pos = prev_var.pos;
                    if ((prev_cigar == VAR_OP_X) && (prev_pos == var.pos)) {
                        skip_ref_base_before_ins = true;
                    } else if ((prev_cigar == VAR_OP_D) && (prev_pos + prev_len > var.pos)) {
                        skip_ref_base_before_ins = true;
                    }
                }
                if (skip_ref_base_before_ins) {
                    if constexpr (WUT_VERBOSE) {
                        spdlog::debug(
                                "[wut][r {}] pushing INS(1) first: skipped ref base at offset {} "
                                "(major pos {}; prev cigar {}; prev pos {}; prev var len {})\n",
                                gck.readID2qn[i_read], offset, majors[offset], prev_cigar, prev_pos,
                                prev_len);
                    }
                } else {
                    if constexpr (WUT_VERBOSE) {
                        spdlog::debug(
                                "[wut][r {}] pushing INS(1) first: filled ref base at offset {} "
                                "(major pos {})\n",
                                gck.readID2qn[i_read], offset, majors[offset]);
                    }
                    if (offset < majors.size()) {
                        expanded_read[offset] = 6;
                        offset++;
                    }
                }
                // (no need to touch i_var_incre)

                // (insertion bases, right-align)
                const uint32_t exp_size = gck.all_ins[read.vars[i_var].pos];
                if (options.right_align_insertions) {
                    const uint32_t space_size = exp_size - (get_qa_t_allele_len(var));
                    for (uint32_t i = 0; i < space_size && offset < majors.size(); i++, offset++) {
                        expanded_read[offset] = 5;
                    }
                    for (uint32_t i = space_size, j = 0; i < exp_size && offset < majors.size();
                         i++, j++) {
                        expanded_read[offset] = var.allele[j] + 1;
                        offset++;
                    }
                } else {  // left-align
                    const uint32_t var_size = get_qa_t_allele_len(var);
                    for (uint32_t i = 0, j = 0; i < var_size && offset < majors.size();
                         i++, j++, offset++) {
                        expanded_read[offset] = var.allele[j] + 1;
                    }
                    for (uint32_t i = var_size; i < exp_size && offset < majors.size();
                         i++, offset++) {
                        expanded_read[offset] = 5;
                    }
                }
                i_var_incre = 1;
            } else {
                spdlog::error("[kdys::{}] r {} has unknown cigar (op {} var pos {})", __func__,
                              gck.readID2qn[i_read], var.allele.back(), var.pos);
            }

            // the matches after it
            if (offset < majors.size()) {
                const uint32_t pos_right = i_var + i_var_incre >= read.vars.size()
                                                   ? read.end_pos
                                                   : read.vars[i_var + i_var_incre].pos;
                if constexpr (WUT_VERBOSE) {
                    spdlog::debug(
                            "[wut][r {}] pushing match: pos {} - {} (read end pos is {}, itvl_end "
                            "is {})\n",
                            gck.readID2qn[i_read], majors[offset], pos_right, read.end_pos,
                            itvl_end);
                }
                for (uint32_t pos = majors[offset]; pos < pos_right; pos++) {
                    if constexpr (WUT_VERBOSE) {
                        spdlog::debug(
                                "[wut][r {}] push match (offset {} pos {}) (ref base should be "
                                "{})\n",
                                gck.readID2qn[i_read], offset, majors[offset],
                                "0ACGT-"[gck.refseq_substring[pos - itvl_start]]);
                    }
                    expanded_read[offset] = 6;
                    offset++;
                    while (offset < minors.size() && minors[offset] != 0) {
                        expanded_read[offset] = 5;
                        offset++;
                    }
                }
            }

            // step forward
            i_var += i_var_incre;
        }

        // We have collected the expanded read.
        // Fill the base quals and dwells.
        const std::vector<uint8_t> &quals = gck.all_reads_quals[i_read];
        const std::vector<int8_t> &dwells = gck.all_dwells[i_read];
        for (uint32_t i = offset0, j_qual = 0, j_dwell = 0;
             i < offset && j_qual < quals.size() && j_dwell < dwells.size(); i++) {
            if (expanded_read[i] != 0 && expanded_read[i] != 5) {
                expanded_read_quals[i] = quals[j_qual];
                expanded_read_dwells[i] = dwells[j_dwell];
                j_qual++;
                j_dwell++;
            } else {
                expanded_read_quals[i] = -1;
                expanded_read_dwells[i] = -1;
            }
        }

        // Copy over to the matrix. Implicit transposition.
        uint32_t insert_offset0 = offset0;
        if (read2matrix_start_pos[i_read] > read.start_pos) {
            insert_offset0 = read2matrix_start_pos[i_read] - itvl_start;
            for (const medaka_feature_matrix_expansion_entry_t &exp : expansions) {
                if (exp.pos >= read2matrix_start_pos[i_read]) {
                    break;
                }
                insert_offset0 += exp.len;
            }
        }

        gen_medaka_feature_matrix_insert_to_matrix(ret, qname2hp, gck, i_read, laneID,
                                                   insert_offset0, offset, expanded_read,
                                                   expanded_read_quals, expanded_read_dwells);

        // cleanup buffers
        std::fill(expanded_read.begin() + offset0, expanded_read.begin() + offset, 0);
        std::fill(expanded_read_quals.begin() + offset0, expanded_read_quals.begin() + offset, 0);
        std::fill(expanded_read_dwells.begin() + offset0, expanded_read_dwells.begin() + offset, 0);
    }

    return ret;
}

MedakaFeatureMatrix gen_medaka_feature_matrix_wrapper(
        dorado::secondary::BamFile &bam_file,
        const std::string &refname,
        uint32_t itvl_start,
        uint32_t itvl_end,
        const std::unordered_map<std::string, int32_t> &qname2hp,
        const MedakaFeatureMatrixOptions &options) {
    // refname and interval start & end are needed only for the
    // coordinates and debugging. They are not used to retrieve ref seq.
    dorado::secondary::BamFileView hf = bam_file.get_view();
    return gen_medaka_feature_matrix(hf, refname, itvl_start, itvl_end, qname2hp, options);
}

static void print_medaka_feature_matrix1(std::ostream &out, const MedakaFeatureMatrix &mfm) {
    out << "#buffer_pos,buffer_reads,num_dtypes,n_pos,n_reads,featlen,mat size,major size,minor "
           "size\n";
    out << mfm.buffer_pos << ',' << mfm.buffer_reads << ',' << mfm.num_dtypes << ',' << mfm.n_pos
        << ',' << mfm.n_reads << ',' << mfm.featlen << ',' << mfm.matrix.size() << ','
        << mfm.major.size() << ',' << mfm.minor.size() << '\n';

    // print boundary read names
    std::vector<std::string> qnames = mfm.read_ids_left;
    int i = 0;
    for (const std::string &qn : qnames) {
        out << "L\t" << i << '\t' << qn << '\n';
        i++;
    }
    qnames = mfm.read_ids_right;
    i = 0;
    for (const std::string &qn : qnames) {
        out << "R\t" << i << '\t' << qn << '\n';
        i++;
    }

    // right now mfm is position-major
    for (int32_t i_pos = 0; i_pos < mfm.buffer_pos; i_pos++) {
        for (int i_feature = 0; i_feature < 7; i_feature++) {
            out << (mfm.major[i_pos] + 1) << '\t' << mfm.minor[i_pos] << '\t';
            for (int32_t i_lane = 0; i_lane < mfm.buffer_reads; i_lane++) {
                if (i_feature == 0) {
                    out << "0ACGT-"[mfm.matrix[i_pos * mfm.buffer_reads * mfm.featlen +
                                               i_lane * mfm.featlen + i_feature]];
                } else {
                    out << std::to_string(mfm.matrix[i_pos * mfm.buffer_reads * mfm.featlen +
                                                     i_lane * mfm.featlen + i_feature]);
                    out << ',';
                }
            }
            out << '\n';
        }
    }
}

void print_medaka_feature_matrix(const std::filesystem::path &out, const MedakaFeatureMatrix &mfm) {
    std::ofstream fp(out);
    print_medaka_feature_matrix1(fp, mfm);
}

std::string print_medaka_feature_matrix(const MedakaFeatureMatrix &mfm) {
    std::ostringstream oss;
    print_medaka_feature_matrix1(oss, mfm);
    return std::move(oss).str();
}

}  // namespace kadayashi
