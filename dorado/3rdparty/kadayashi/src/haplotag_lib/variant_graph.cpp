#include "variant_graph.h"

#include "sequence_utility.h"

#include <algorithm>
#include <array>
#include <cstdio>

namespace kadayashi {

constexpr float HAP_TAG_MAX_CNFLCT_RATIO = 0.5f;

namespace {

uint32_t variant_graph_get_edge_val(const variant_graph_t &vg,
                                    const int i,
                                    const uint8_t H1,
                                    const uint8_t H2) {
    return vg.edges[i].counts[(H1 << 1) | H2];
}
uint32_t variant_graph_get_edge_val2(const variant_graph_t &vg, const int i, const uint8_t comb) {
    return vg.edges[i].counts[comb];
}
uint32_t variant_graph_get_node_val2(const variant_graph_t &vg, const int i, const uint8_t comb) {
    return vg.nodes[i].scores[comb];
}
uint32_t &variant_graph_get_node_val2_ref(variant_graph_t &vg, const int i, const uint8_t comb) {
    return vg.nodes[i].scores[comb];
}

void variant_graph_update_varhp_given_read(const chunk_t &ck,
                                           std::vector<std::array<float, 3>> &varhps,
                                           const int readID,
                                           const uint8_t hp) {
    const read_t &r = ck.reads[readID];
    for (size_t i_var = 0; i_var < r.vars.size(); i_var++) {
        const uint32_t idx_var = r.vars[i_var].var_idx;
        if (r.vars[i_var].allele_idx > 1) {
            continue;
        }
        varhps[idx_var][hp ^ r.vars[i_var].allele_idx] += 1;
        varhps[idx_var][2] += 1;
    }
}

struct infer_readhp_stat_t {
    float score_best = 0.0f;
    int updated_best = 0;
    uint8_t hp_best = HAPTAG_UNPHASED;
};
infer_readhp_stat_t variant_graph_infer_readhp_given_vars(
        const chunk_t &ck,
        const std::vector<std::array<float, 3>> &varhps,
        const uint32_t readID) {
    const read_t &r = ck.reads[readID];
    float hp0 = 0.0f;
    float hp1 = 0.0f;
    int updated = 0;
    for (size_t i_var = 0; i_var < r.vars.size(); i_var++) {
        const uint32_t idx_var = r.vars[i_var].var_idx;
        const uint32_t idx_allele = r.vars[i_var].allele_idx;
        if (idx_allele > 1) {
            continue;
        }
        if (varhps[idx_var][2] < 0.1f) {
            continue;  // accumulate count is zero
        }
        if (idx_allele == 0) {
            hp0 += varhps[idx_var][0] / varhps[idx_var][2];
            hp1 += varhps[idx_var][1] / varhps[idx_var][2];
        } else {
            hp0 += varhps[idx_var][1] / varhps[idx_var][2];
            hp1 += varhps[idx_var][0] / varhps[idx_var][2];
        }
        updated++;
    }

    if (updated) {
        return hp0 > hp1 ? infer_readhp_stat_t{.score_best = hp0,
                                               .updated_best = updated,
                                               .hp_best = 0}
                         : infer_readhp_stat_t{
                                   .score_best = hp1, .updated_best = updated, .hp_best = 1};
    }
    return infer_readhp_stat_t{.score_best = 0.0f, .updated_best = 0, .hp_best = HAPTAG_UNPHASED};
}

infer_readhp_stat_t variant_graph_infer_readhp_given_vars_ht(
        chunk_t &ck,
        std::unordered_map<uint32_t, std::array<float, 3>> &pos2counter,
        uint32_t readID,
        int debug_print) {
    read_t &r = ck.reads[readID];
    float hp0 = 0.0f;
    float hp1 = 0.0f;
    int updated = 0;
    for (size_t i_var = 0; i_var < r.vars.size(); i_var++) {
        if (r.vars[i_var].is_used != TA_STAT_ACCEPTED) {
            continue;
        }
        const uint32_t pos = r.vars[i_var].pos;
        const uint32_t idx_allele = r.vars[i_var].allele_idx;
        if (idx_allele > 1) {
            continue;
        }
        if (pos2counter[pos][2] < 0.1f) {
            continue;  // accumulate count is zero
        }

        if (idx_allele == 0) {
            hp0 += pos2counter[pos][0] / pos2counter[pos][2];
            hp1 += pos2counter[pos][1] / pos2counter[pos][2];
        } else {
            hp0 += pos2counter[pos][1] / pos2counter[pos][2];
            hp1 += pos2counter[pos][0] / pos2counter[pos][2];
        }
        if (debug_print) {
            fprintf(stderr, "[dbg::%s]   try qn %s : pos %d hp0=%.1f hp1=%.1f\n", __func__,
                    ck.qnames[readID].c_str(), (int)pos, hp0, hp1);
        }

        updated++;
    }

    infer_readhp_stat_t ret;
    if (updated) {
        if (hp0 > hp1) {
            ret.score_best = hp0;
            ret.updated_best = updated;
            ret.hp_best = 0;
        } else {
            ret.score_best = hp1;
            ret.updated_best = updated;
            ret.hp_best = 1;
        }
    }
    return ret;
}

void variant_graph_update_varhp_given_read_ht(
        chunk_t &ck,
        std::unordered_map<uint32_t, std::array<float, 3>>
                &pos2counter,  // ref allele's: hap0, hap1, hap0+hap1
        int readID,
        uint8_t hp) {
    read_t &r = ck.reads[readID];
    for (size_t i_var = 0; i_var < r.vars.size(); i_var++) {
        if (r.vars[i_var].allele_idx > 1) {
            continue;
        }
        const uint32_t pos = r.vars[i_var].pos;
        if (pos2counter.find(pos) == pos2counter.end()) {
            pos2counter[pos] = {0.0f, 0.0f, 0.0f};
        }
        pos2counter[pos][hp ^ r.vars[i_var].allele_idx] += 1;
        pos2counter[pos][2] += 1;
    }
}

int normalize_readtaggings_count(const std::vector<uint8_t> &d1, const std::vector<uint8_t> &d2) {
    // Returns number of reads that can be used to anchor the read phasing of
    // a later iteration (d2) to that of a previous one (d1).
    // Returns -1 when input is not valid.

    int comparable = 0;

    if (d1.size() != d2.size()) {  // fallback case for non-debug run; should not happen
        return -1;
    }
    for (size_t i = 0; i < d1.size(); i++) {
        if (d1[i] != HAPTAG_UNPHASED && d2[i] != HAPTAG_UNPHASED) {
            comparable++;
        }
    }
    return comparable;
}

std::pair<int, int> normalize_readtaggings1(const std::vector<uint8_t> &d1,
                                            std::vector<uint8_t> &d2) {
    if (d1.size() != d2.size()) {  // fallback case for non-debug run; should not happen
        return std::make_pair(0, 0);
    }
    const size_t l = d1.size();

    int score_raw = 0;
    int score_flip = 0;
    for (size_t i = 0; i < l; i++) {
        if (d1[i] != HAPTAG_UNPHASED && d2[i] != HAPTAG_UNPHASED) {
            if (d1[i] == d2[i]) {
                score_raw += 1;
            } else {
                score_flip += 1;
            }
        }
    }
    if (score_flip > score_raw) {
        for (size_t i = 0; i < l; i++) {
            if (d2[i] != HAPTAG_UNPHASED) {
                d2[i] ^= 1;
            }
        }
    }
    return std::make_pair(score_raw, score_flip);
}

bool normalize_readtaggings(std::vector<std::vector<uint8_t>> &data) {
    // return true when ok

    constexpr int DEBUG_PRINT = 0;
    if (data.size() <= 1) {
        return false;
    }
    for (int64_t i = 0; i < std::ssize(data) - 1; i++) {
        if (data[i].size() != data[i + 1].size()) {
            if constexpr (DEBUG_PRINT) {
                fprintf(stderr,
                        "[E::%s] data entries have at least 1 pair of unequal lengths, should not "
                        "happen.\n",
                        __func__);
            }
            return false;
        }
    }

    if constexpr (DEBUG_PRINT) {
        fprintf(stderr, "[dbg::%s]", __func__);
        for (size_t i = 0; i < data[0].size(); i++) {
            fprintf(stderr, " %d", static_cast<int32_t>(data[0][i]));
        }
        fprintf(stderr, "\n");
    }

    for (size_t i = 1; i < data.size(); i++) {
        int i_ref = 0;
        int count_ref = 0;
        for (size_t j = 0; j < i; j++) {  // find a previous one as the reference for flipping
            const int comparable = normalize_readtaggings_count(data[j], data[i]);
            if (comparable < 0) {
                if constexpr (DEBUG_PRINT) {
                    fprintf(stderr,
                            "[E::%s] entries have unequal lengths, should not happen, check code\n",
                            __func__);
                }
                continue;
            }
            if (comparable > count_ref) {
                count_ref = comparable;
                i_ref = static_cast<int>(j);
            }
        }
        if (count_ref == 0) {
            continue;
        }

        const std::pair<int, int> counts = normalize_readtaggings1(data[i_ref], data[i]);

        if constexpr (DEBUG_PRINT) {
            fprintf(stderr, "[dbg::%s] <%d %d>\t", __func__, counts.first, counts.second);
            for (size_t k = 0; k < data[i].size(); k++) {
                fprintf(stderr, " %d", static_cast<int32_t>(data[i][k]));
            }
            fprintf(stderr, "\n");
        }
    }
    return true;
}

int variant_graph_pos_score_diff_of_top_two(const variant_graph_t &vg, const int var_idx) {
    std::array<uint32_t, 4> tmpcounter = {0};
    for (int tmpi = 0; tmpi < 4; tmpi++) {
        tmpcounter[tmpi] = vg.nodes[var_idx].scores[tmpi];
    }
    std::sort(tmpcounter.begin(), tmpcounter.end(), std::greater<>());
    return tmpcounter[0] - tmpcounter[1];
}

void variant_graph_get_edge_values(chunk_t &ck,
                                   uint32_t varID1,
                                   uint32_t varID2,
                                   uint32_t counter[4]) {
    for (int i = 0; i < 4; i++) {
        counter[i] = 0;
    }
    assert(varID1 < varID2);
    for (const read_t &r : ck.reads) {
        if (!r.vars.empty()) {
            int i1 = -1;
            int i2 = -1;
            for (int i = 0; i < (static_cast<int>(r.vars.size()) - 1); i++) {
                assert(r.vars[i].is_used);
                if (r.vars[i].var_idx == varID1) {
                    i1 = i;
                }
                if (r.vars[i].var_idx == varID2) {
                    i2 = i;
                    break;
                }
            }
            if (i1 != -1 && i2 != -1) {
                i1 = r.vars[i1].allele_idx;
                i2 = r.vars[i2].allele_idx;
                if ((i1 != 0 && i1 != 1) || (i2 != 0 && i2 != 1)) {
                    fprintf(stderr,
                            "[E::%s] 2-allele diploid sancheck failed, impossible, check code. "
                            "Results may be wrong.\n",
                            __func__);
                    continue;
                }
                counter[i1 << 1 | i2]++;
            }
        }
    }
}

int variant_graph_init_scores_for_a_location(chunk_t &ck, const uint32_t var_idx, bool do_wipeout) {
    // return 1 when re-init was a hard one
    constexpr int DEBUG_PRINT = 0;
    int ret = -1;
    variant_graph_t &vg = ck.vg;
    const uint32_t pos = ck.varcalls[var_idx].pos;
    uint32_t counter[4] = {0, 0, 0, 0};

    if (var_idx != 0 && !do_wipeout) {
        // reuse values from a previous position
        int i = 0;
        int diff = 5;
        for (int k = var_idx - 1; k > 0; k--) {
            const uint32_t pos2 = ck.varcalls[k].pos;
            if constexpr (DEBUG_PRINT) {
                fprintf(stderr, "trying pos_resume %d (k=%d)\n", pos2, k);
            }
            if (pos - pos2 > 50000) {
                break;  // too far
            }
            if (!vg.next_link_is_broken[k - 1]) {
                if constexpr (DEBUG_PRINT) {
                    fprintf(stderr, "trying pos_resume %d (k=%d) checkpoint 1\n", pos2, k);
                }
                if (pos - pos2 > 300 && (pos - pos2 < 5000 ||
                                         i == 0)) {  // search within 5kb or until finally found one
                    const int tmpdiff = variant_graph_pos_score_diff_of_top_two(vg, k);
                    if (tmpdiff > diff) {
                        diff = tmpdiff;
                        i = k;
                    }
                }
            }
        }
        if (i > 0) {
            if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
                fprintf(stderr, "[dbg::%s] *maybe* re-init pos %d using pos %d\n", __func__,
                        (int)pos, (int)ck.varcalls[i].pos);
                for (uint8_t comb = 0; comb < 4; comb++) {
                    fprintf(stderr, "[dbg::%s] original %d : %d\n", __func__, i,
                            static_cast<int>(variant_graph_get_node_val2(vg, var_idx, comb)));
                }
            }
            variant_graph_get_edge_values(ck, static_cast<uint32_t>(i),
                                          static_cast<uint32_t>(var_idx), counter);

            // we forced to grab at least one resume point,
            // need to check if there's any read supporting the connection.
            // if not, resort to hard init as this is probably a
            // real phasing break.
            constexpr int MIN_SUPPORTING_READS = 3;
            if (std::all_of(std::begin(counter), std::end(counter),
                            [](uint32_t val) { return val < MIN_SUPPORTING_READS; })) {
                do_wipeout = true;
            } else {
                if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
                    fprintf(stderr,
                            "[dbg::%s] *maybe* not wiping; edge counts are %d %d %d %d; var idx "
                            "are %d and %d\n",
                            __func__, counter[0], counter[1], counter[2], counter[3], i, var_idx);
                }

                std::array<uint32_t, 4> bests = {0};
                vgnode_t *n1 = &vg.nodes[var_idx];
                for (uint8_t comb = 0; comb < 4; comb++) {
                    n1->scores[comb] = vg.nodes[i].scores[comb];
                    bests[comb] = n1->scores[comb];
                }
                int best_i = 0;
                max_of_u32_arr(bests, &best_i, nullptr);
                n1->best_score_i = best_i;

                if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
                    for (uint8_t comb = 0; comb < 4; comb++) {
                        fprintf(stderr, "[dbg::%s] new %d : %d\n", __func__, static_cast<int>(comb),
                                static_cast<int>(variant_graph_get_node_val2(vg, var_idx, comb)));
                    }
                }
            }
        } else {
            if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
                fprintf(stderr, "[dbg::%s] tried but failed to find resume point\n", __func__);
            }
            do_wipeout = true;  // did not find a good resume point, will hard re-init
        }
    }

    if (var_idx == 0 || do_wipeout) {
        ret = 1;
        if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr, "[dbg::%s] hard re-init for pos %d\n", __func__, (int)pos);
        }
        for (int i = 0; i < 4; i++) {
            counter[i] = 0;
        }
        for (read_t &r : ck.reads) {
            if (r.start_pos > pos) {
                break;
            }  // reads are loaded from sorted bam, safe to break here
            if (r.end_pos <= pos) {
                continue;
            }
            for (size_t i = 0; i < r.vars.size(); i++) {
                if (r.vars[i].var_idx == var_idx) {
                    if (r.vars[i].allele_idx == 0) {
                        counter[0]++;
                    } else if (r.vars[i].allele_idx == 1) {
                        counter[3]++;
                    }
                    break;
                }
            }
        }
        variant_graph_get_node_val2_ref(vg, var_idx, 0) = counter[0];
        variant_graph_get_node_val2_ref(vg, var_idx, 1) = counter[0] + counter[3];
        variant_graph_get_node_val2_ref(vg, var_idx, 2) = 0;
        variant_graph_get_node_val2_ref(vg, var_idx, 3) = counter[3];
    } else {
        ret = 0;
    }

    std::array<uint32_t, 4> tmp = {0};
    for (uint8_t comb = 0; comb < 4; comb++) {
        tmp[comb] = variant_graph_get_node_val2(vg, var_idx, comb);
    }
    int best_i = 0;
    max_of_u32_arr(tmp, &best_i, nullptr);
    vg.nodes[var_idx].best_score_i = best_i;
    for (int i = 0; i < 4; i++) {
        vg.nodes[var_idx].scores_source[i] = 4;  // sentinel
    }
    assert(ret >= 0);
    return ret;
}

void variant_graph_propogate_one_step(chunk_t &ck, int *i_prev_, const int i_self) {
    // note: we redo initialization at cov dropouts rather than
    // let backtracing figure out these phasing breakpoints.
    // The breakpoints are stored in vg. When haptagging
    // a read, we will not mix evidences from different phase blocks.
    constexpr int DEBUG_PRINT = 0;
    assert(i_self > 0);

    variant_graph_t &vg = ck.vg;
    int i_prev = *i_prev_;
    vgnode_t *n1 = &vg.nodes[i_self];

    std::array<uint32_t, 4> bests = {0};
    if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
        const std::string a0 = nt4seq2seq(ck.varcalls[i_self].alleles[0]);
        const std::string a1 = nt4seq2seq(ck.varcalls[i_self].alleles[1]);
        fprintf(stderr, "[dbg::%s] i_self=%d (pos=%d a1=%s a2=%s):\n", __func__, i_self,
                ck.varcalls[i_self].pos, a0.c_str(), a1.c_str());
    }

    // check whether we have a coverage dropout
    for (uint8_t i = 0; i < 4; i++) {
        bests[i] = variant_graph_get_edge_val2(vg, i_prev, i);
    }
    uint32_t best;
    max_of_u32_arr(bests, nullptr, &best);
    if (best < 3) {  // less than 3 reads support any combination, spot is
                     // a coverage dropout, redo initialization.
        int reinit_failed = variant_graph_init_scores_for_a_location(ck, i_self, false);
        if (reinit_failed) {
            vg.next_link_is_broken[i_prev] = 1;
            vg.has_breakpoints = 1;
        }
        if (variant_graph_pos_score_diff_of_top_two(vg, i_self) <= 5) {
            vg.next_link_is_broken[i_self] = 1;
        }
        *i_prev_ = i_self;
        if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr, "[dbg::%s]    ! phasing broke at %d (coverage dropout)\n", __func__,
                    ck.varcalls[i_self].pos);
        }
    }

    for (uint8_t self_combo = 0; self_combo < 4; self_combo++) {
        std::array<uint32_t, 4> score = {0};
        for (uint8_t prev_combo = 0; prev_combo < 4; prev_combo++) {
            int both_hom =
                    ((self_combo == 0 || self_combo == 3) && (prev_combo == 0 || prev_combo == 3));
            int s1 = variant_graph_get_node_val2(vg, i_prev, prev_combo);
            int s2 = variant_graph_get_edge_val(vg, i_prev, prev_combo >> 1, self_combo >> 1);
            int s3 = both_hom ? 0
                              : variant_graph_get_edge_val(vg, i_prev, prev_combo & 1,
                                                           self_combo & 1);

            score[prev_combo] = s1 + s2 + s3;
            if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
                fprintf(stderr,
                        "[dbg::%s]  self combo %d, %d + %d + %d = %d(i_prev=%d; key1=%d key2=%d)\n",
                        __func__, self_combo, s1, s2, s3, score[prev_combo], i_prev,
                        (prev_combo >> 1) << 1 | (self_combo >> 1),
                        (prev_combo & 1) << 1 | (self_combo & 1));
            }
        }
        int source = 0;
        uint32_t best_score;
        max_of_u32_arr(score, &source, &best_score);
        bests[self_combo] = best_score;
        n1->scores[self_combo] = bests[self_combo];
        n1->scores_source[self_combo] = static_cast<uint8_t>(source);
    }
    int best_i = 0;
    max_of_u32_arr(bests, &best_i, nullptr);
    n1->best_score_i = best_i;

    // another check: if phasing is broken, redo init for self
    if (best_i == 0 || best_i == 3) {  // decision was hom
        int reinit_failed = variant_graph_init_scores_for_a_location(ck, i_self, false);
        if (reinit_failed) {
            vg.next_link_is_broken[i_prev] = 1;
            vg.has_breakpoints = 1;
        }
        if (variant_graph_pos_score_diff_of_top_two(vg, i_self) <= 5) {
            vg.next_link_is_broken[i_self] = 1;
        }
        if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr, "[dbg::%s]    ! phasing broke at %d (hom decision = %d)\n", __func__,
                    ck.varcalls[i_self].pos, best_i);
        }
    }

    *i_prev_ = i_self;
    if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
        fprintf(stderr, "[dbg::%s]    best i: %d (bests: %d %d %d %d)\n", __func__, best_i,
                bests[0], bests[1], bests[2], bests[3]);
    }
}

}  // namespace

/*** 2-allele diploid local variant graph ***/
bool variant_graph_gen(chunk_t &ck) {
    if (!ck.is_valid || std::empty(ck.varcalls)) {
        return false;
    }

    variant_graph_t &vg = ck.vg;

    vg.n_vars = static_cast<uint32_t>(std::size(ck.varcalls));
    vg.nodes.resize(vg.n_vars);
    vg.edges.resize(vg.n_vars - 1);
    vg.next_link_is_broken.resize(vg.n_vars);

    // fill in nodes
    for (size_t i = 0; i < ck.varcalls.size(); i++) {
        assert(ck.varcalls[i].is_used);
        vg.nodes[i] = {.ID = static_cast<uint32_t>(i)};
        vg.nodes[i].del = 0;
    }

    // fill in edges
    for (int64_t i_read = 0; i_read < std::ssize(ck.reads); i_read++) {
        const read_t &r = ck.reads[i_read];
        if (!r.vars.empty()) {
            for (int64_t i = 0; i < std::ssize(r.vars) - 1; i++) {
                assert(r.vars[i].is_used);
                const uint32_t varID1 = r.vars[i].var_idx;
                const uint32_t varID2 = r.vars[i + 1].var_idx;
                if (varID2 - varID1 != 1) {
                    continue;
                }
                const uint8_t i1 = static_cast<uint8_t>(r.vars[i].allele_idx);
                const uint8_t i2 = static_cast<uint8_t>(r.vars[i + 1].allele_idx);
                if (((i1 != 0) && (i1 != 1)) || ((i2 != 0) && (i2 != 1))) {
                    fprintf(stderr,
                            "[E::%s] this impl is 2-allele diploid, sancheck failed; should not "
                            "happen here, check code. Not incrementing edge weight\n",
                            __func__);
                } else {
                    vg.edges[varID1].counts[i1 << 1 | i2]++;
                }
            }
        }
    }

    // init the first node
    variant_graph_init_scores_for_a_location(ck, 0, true);

    return true;
}

void variant_graph_propogate(chunk_t &ck) {
    // Fill in scores for nodes.
    int i_prev = 0;
    for (uint32_t i = 1; i < ck.vg.n_vars; i++) {
        variant_graph_propogate_one_step(ck, &i_prev, i);
    }
}

bool variant_graph_check_if_phasing_succeeded(const chunk_t &ck) {
    constexpr int VERBOSE = 0;
    if (!ck.is_valid || ck.vg.n_vars == 0) {
        return false;
    }

    // return: true if fully phased, false if phase is broken at somewhere
    // If we ever have a best score that suggests homozygous assignment,
    // the phasing is broken.
    bool is_ok = true;
    int source;
    for (int i_pos = ck.vg.n_vars - 1; i_pos > 0; i_pos--) {
        if (ck.vg.nodes[i_pos].del) {
            continue;
        }
        source = ck.vg.nodes[i_pos].scores_source[ck.vg.nodes[i_pos].best_score_i];
        if (source == 0 || source == 3) {
            is_ok = false;
            if constexpr (VERBOSE) {
                fprintf(stderr, "[dbg::%s] phasing broke at pos %d \n", __func__,
                        ck.varcalls[i_pos].pos);
            } else {
                return false;
            }
        }
    }
    return is_ok;
}

void variant_graph_haptag_reads(chunk_t &ck) {
    // 2-allele diploid, dvr method
    // Given phased variants, assign haptags to reads.
    constexpr int DEBUG_PRINT = 0;
    variant_graph_t &vg = ck.vg;
    int sancheck_cnt[5] = {
            0, 0,
            0,  // no variant
            0,  // ambiguous
            0,  // unphaseddue to conflict
    };
    uint32_t var_i_start = 0;
    uint32_t var_i_end = vg.n_vars;
    std::vector<uint64_t> buf;

    for (size_t i_read = 0; i_read < ck.reads.size(); i_read++) {
        read_t *r = &ck.reads[i_read];
        if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr, "[dbg::%s] saw qn %s\n", __func__, ck.qnames[i_read].c_str());
        }

        if (r->vars.empty()) {
            if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
                fprintf(stderr, "[dbg::%s] skip %s (no var)\n", __func__,
                        ck.qnames[i_read].c_str());
            }
            r->hp = HAPTAG_UNPHASED;
            sancheck_cnt[2]++;
            continue;
        }

        if (vg.has_breakpoints) {
            // find the largest phase block overlapping
            // with the read, and use only its variants.
            buf.clear();
            uint64_t cnt = 0;
            uint32_t start = r->vars[0].var_idx;
            int j = -1;
            int broken = 0;
            for (size_t i = 0; i < r->vars.size(); i++) {
                int idx = r->vars[i].var_idx;
                if (j == -1) {
                    j = idx;
                }
                while (j < idx + 1) {
                    if (vg.next_link_is_broken[j]) {
                        broken = 1;
                        break;
                    }
                    cnt++;
                    j++;
                }
                if (broken) {
                    buf.push_back(cnt << 32 | start);
                    start = idx + 1;
                    broken = 0;
                    cnt = 0;
                }
            }
            if (cnt > 0) {
                buf.push_back(cnt << 32 | start);
            }
            // (do we have any variants?)
            if (buf.empty()) {
                if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
                    fprintf(stderr, "[dbg::%s] skip %s (no intersecting var)\n", __func__,
                            ck.qnames[i_read].c_str());
                }
                r->hp = HAPTAG_UNPHASED;
                sancheck_cnt[2]++;
                continue;
            }

            // (get largest block)
            std::sort(buf.begin(), buf.end());
            var_i_start = static_cast<uint32_t>(buf.back());
            for (uint32_t i = var_i_start; i < vg.n_vars; i++) {
                var_i_end = i + 1;
                if (vg.next_link_is_broken[i]) {
                    break;
                }
            }
            if (var_i_end <= var_i_start) {
                var_i_end = var_i_start + 1;  // interval specified as [)
            }
            if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
                fprintf(stderr,
                        "[dbg::%s] (now using [s=%d e=%d] (%d blocks available; read has %d vars):",
                        __func__, var_i_start, var_i_end, (int)buf.size(), (int)r->vars.size());
                for (uint32_t i = var_i_start; i < var_i_end; i++) {
                    fprintf(stderr, "%d, ", ck.varcalls[i].pos);
                }
                fprintf(stderr, "\n");
            }
        }

        int votes[2] = {0, 0};
        int veto = 0;
        for (int i = 0; i < static_cast<int>(r->vars.size()); i++) {
            const int i_pos = i;
            if (r->vars[i].var_idx < var_i_start) {
                continue;
            }
            if (r->vars[i].var_idx >= var_i_end) {
                continue;
            }
            if (vg.nodes[r->vars[i].var_idx].del) {
                uint32_t pos = ck.varcalls[r->vars[i].var_idx].pos;
                if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
                    fprintf(stderr, "[dbg::%s] veto at pos=%d\n", __func__, pos);
                }
                veto++;
                continue;
            }
            const uint8_t combo = static_cast<uint8_t>(vg.nodes[r->vars[i].var_idx].best_score_i);
            if (combo == 0 || combo == 3) {
                continue;  // position's best phase is a hom
            }
            int idx = r->vars[i_pos].allele_idx;
            if (idx == (combo >> 1)) {
                if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
                    fprintf(stderr, "[dbg::%s]    %s pos=%d hap 0 (idx=%d combo=%d)\n", __func__,
                            ck.qnames[i_read].c_str(), r->vars[i_pos].pos, idx, combo);
                }
                votes[0]++;
            } else if (idx == (combo & 1)) {
                if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
                    fprintf(stderr, "[dbg::%s]    %s pos=%d hap 1 (idx=%d combo=%d)\n", __func__,
                            ck.qnames[i_read].c_str(), r->vars[i_pos].pos, idx, combo);
                }
                votes[1]++;
            } else {
                fprintf(stderr,
                        "[E::%s] %s qn=%d impossible (combo=%d idx=%d), check code. This read will "
                        "be untagged.\n",
                        __func__, ck.qnames[i_read].c_str(), r->vars[i_pos].pos, combo, idx);
                votes[0] = 0;
                votes[1] = 0;
                break;
            }
        }
        if (votes[0] > votes[1] && votes[0] > veto) {
            if (static_cast<float>(votes[1]) / static_cast<float>(votes[0]) <=
                HAP_TAG_MAX_CNFLCT_RATIO) {
                r->hp = 0;
                sancheck_cnt[0]++;
            } else {
                r->hp = HAPTAG_UNPHASED;
                sancheck_cnt[4]++;
            }
        } else if (votes[1] > votes[0] && votes[1] > veto) {
            if (static_cast<float>(votes[0]) / static_cast<float>(votes[1]) <=
                HAP_TAG_MAX_CNFLCT_RATIO) {
                r->hp = 1;
                sancheck_cnt[1]++;
            } else {
                r->hp = HAPTAG_UNPHASED;
                sancheck_cnt[4]++;
            }
        } else {
            r->hp = HAPTAG_UNPHASED;
            sancheck_cnt[3]++;
        }
        r->votes_diploid[0] = votes[0];
        r->votes_diploid[1] = votes[1];

        if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr, "[dbg::%s] qname %s vote0=%d vote1=%d veto=%d => hp=%d\n", __func__,
                    ck.qnames[i_read].c_str(), votes[0], votes[1], veto, r->hp);
        }
    }
    if (DEBUG_LOCAL_HAPLOTAGGING) {
        fprintf(stderr,
                "[M::%s] n_reads %d, hap0=%d hap1=%d no_variant=%d ambiguous=%d "
                "unphased_due_conflict=%d\n",
                __func__, (int)ck.reads.size(), sancheck_cnt[0], sancheck_cnt[1], sancheck_cnt[2],
                sancheck_cnt[3], sancheck_cnt[4]);
    }
}

void normalize_readtaggings_ht(std::vector<std::unordered_map<uint32_t, uint8_t>> &arr_read2hp,
                               std::unordered_map<uint32_t, uint8_t> &breakpoint_reads,
                               const chunk_t &ck) {
    constexpr int DEBUG_PRINT = 0;
    if (arr_read2hp.size() <= 1) {
        if constexpr (DEBUG_PRINT) {
            fprintf(stderr, "[M::%s] nothing done\n", __func__);
        }
        return;
    }

    constexpr int MAX_NEEDED_SUCCESS = 10;
    constexpr int MIN_COMPARABLE_VARIANTS = 1;
    constexpr int MAX_CONSECUTIVE_FAILS = 10;

    int j_cutoff = 0;
    for (int i = 1; i < std::ssize(arr_read2hp); i++) {
        auto &ht_self = arr_read2hp[i];
        int n_consecutive_fails = 0;
        int n_success = 0;

        int count_ref = -1;
        int i_ref = 0;
        int best_cis = 0;
        int best_trans = 0;
        for (int j = i - 1; j >= j_cutoff; j--) {  // find a previous one as the reference
            auto &ht_ref = arr_read2hp[j];
            int n_comparable = 0;
            int cis = 0;
            int trans = 0;
            for (const auto &[readID, hp] : ht_self) {
                const auto it = ht_ref.find(readID);
                if (hp != HAPTAG_UNPHASED && it != ht_ref.cend() && it->second != HAPTAG_UNPHASED) {
                    n_comparable++;
                    if (ht_ref[readID] == hp) {
                        cis++;
                    } else {
                        trans++;
                    }
                }
            }
            if (n_comparable > count_ref) {
                count_ref = n_comparable;
                i_ref = j;
                best_cis = cis;
                best_trans = trans;
            }

            if (n_comparable < MIN_COMPARABLE_VARIANTS) {
                n_consecutive_fails++;
                if (n_consecutive_fails > MAX_CONSECUTIVE_FAILS) {
                    break;
                }
            } else {
                n_consecutive_fails = 0;
                n_success += 1;
            }

            if (n_success > MAX_NEEDED_SUCCESS) {
                break;
            }
        }

        if (n_success == 0) {  // has a phasing breakpoint
            // record the left most variant
            // TODO ^^^ this doesn't handle indels well
            uint32_t firstreadID = std::numeric_limits<uint32_t>::max();
            for (auto &[readID, hp] : ht_self) {
                if (hp != HAPTAG_UNPHASED && readID < firstreadID) {
                    firstreadID = readID;
                }
            }
            if (firstreadID != std::numeric_limits<uint32_t>::max()) {
                breakpoint_reads[firstreadID] = 1;
            } else {
                if constexpr (DEBUG_PRINT) {
                    fprintf(stderr,
                            "[E::%s] should not happen: failed to get first read ID at phasing "
                            "breakpoint? Giving up haptag normalization.\n",
                            __func__);
                }
                return;
            }

            if constexpr (DEBUG_PRINT) {
                fprintf(stderr, "[dbg::%s] no change for iter#%d (sample qn: ", __func__, i);
                uint32_t sampleqID = std::numeric_limits<uint32_t>::max();
                uint32_t left_pos = std::numeric_limits<uint32_t>::max();
                uint32_t right_pos = 0;
                for (auto &[qID, _] : ht_self) {
                    if (ck.reads[qID].start_pos < left_pos) {
                        left_pos = ck.reads[qID].start_pos;
                    }
                    if (ck.reads[qID].end_pos >= right_pos) {
                        right_pos = ck.reads[qID].end_pos;
                    }
                    if (sampleqID == std::numeric_limits<uint32_t>::max()) {
                        sampleqID = qID;
                    }
                }
                fprintf(stderr, "%s , pos is : %d-%d)\n", ck.qnames[sampleqID].c_str(),
                        (int)left_pos, (int)right_pos);
            }

            j_cutoff = i;
            continue;
        }

        if (best_trans > best_cis) {  // flip self
            if constexpr (DEBUG_PRINT) {
                fprintf(stderr,
                        "[dbg::%s] flip iter#%d (ref: iter#%d, n_comparable=%d, sample qn: ",
                        __func__, i, i_ref, count_ref);
                uint32_t sampleqID = std::numeric_limits<uint32_t>::max();
                uint32_t left_pos = std::numeric_limits<uint32_t>::max();
                uint32_t right_pos = 0;
                for (auto &[qID, _] : ht_self) {
                    if (ck.reads[qID].start_pos < left_pos) {
                        left_pos = ck.reads[qID].start_pos;
                    }
                    if (ck.reads[qID].end_pos >= right_pos) {
                        right_pos = ck.reads[qID].end_pos;
                    }
                    if (sampleqID == std::numeric_limits<uint32_t>::max()) {
                        sampleqID = qID;
                    }
                }
                fprintf(stderr, "%s , pos is : %d-%d)\n", ck.qnames[sampleqID].c_str(),
                        (int)left_pos, (int)right_pos);
            }
            for (auto &[readID, hp] : arr_read2hp[i]) {
                if (hp != HAPTAG_UNPHASED) {
                    hp ^= 1;
                }
            }
        }
    }
}

void variant_graph_do_simple_haptag(chunk_t &ck, const uint32_t n_iter_requested) {
    constexpr int DEBUG_PRINT = 0;

    uint32_t n_iter = n_iter_requested;
    if (n_iter == 0 || n_iter >= ck.reads.size()) {
        n_iter = static_cast<int>(ck.reads.size());
    }

    std::vector<std::vector<uint8_t>> results;
    results.reserve(n_iter);

    const uint32_t stride = static_cast<int>(ck.reads.size()) / n_iter;
    uint32_t prev_seedID = std::numeric_limits<uint32_t>::max();

    for (uint32_t i_iter = 0; i_iter < n_iter; i_iter++) {
        // used the read with the most number of phasing variants within
        // the current bin
        uint32_t max_var = 0;
        uint32_t i_max_var = i_iter * stride;
        for (uint32_t j = i_iter * stride; j < (i_iter + 1) * stride; j++) {
            uint32_t n_valid_vars = 0;
            for (qa_t &_ : ck.reads[j].vars) {
                if (ck.varcalls[_.var_idx].is_used == TA_STAT_ACCEPTED) {
                    n_valid_vars++;
                }
            }
            if (n_valid_vars > max_var) {
                max_var = n_valid_vars;
                i_max_var = j;
            }
        }
        const uint32_t seedreadID = i_max_var;

        if (max_var == 0 || seedreadID == prev_seedID) {
            continue;
        }
        prev_seedID = seedreadID;

        if constexpr (DEBUG_PRINT) {
            fprintf(stderr,
                    "[dbg::%s] collected a seed (iter# %d/%d), qn %s, range %s:%d-%d, max_var = "
                    "%d var_size=%d\n",
                    __func__, (int)(i_iter), n_iter, ck.qnames[i_max_var].c_str(),
                    ck.refname.c_str(), (int)ck.reads[i_max_var].start_pos,
                    (int)ck.reads[i_max_var].end_pos, max_var,
                    (int)ck.reads[i_max_var].vars.size());
        }

        std::vector<uint8_t> readhps = variant_graph_do_simple_haptag1(ck, seedreadID);
        if (readhps.empty()) {
            if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
                fprintf(stderr,
                        "[E::%s] seedreadID >= total number of reads, should not happen, check "
                        "code\n",
                        __func__);
            }
        } else {
            results.push_back(std::move(readhps));
        }
    }

    const bool norm_ok = normalize_readtaggings(results);
    if (!norm_ok) {
        if constexpr (DEBUG_PRINT) {
            fprintf(stderr, "[E::%s] normalization of read haptags failed\n", __func__);
        }
    } else {
        for (uint32_t i_read = 0; i_read < ck.reads.size(); i_read++) {
            float cnt[3] = {0, 0, 0};
            for (const auto &result : results) {
                if (result[i_read] == HAPTAG_UNPHASED) {
                    cnt[2] += 1;
                } else {
                    cnt[result[i_read]] += 1;
                }
            }

            if ((cnt[0] > 3 && cnt[1] > 3) || (cnt[0] + cnt[1] < 0.5f)) {
                ck.reads[i_read].hp = HAPTAG_UNPHASED;
            } else {
                if (cnt[0] > cnt[1]) {
                    ck.reads[i_read].hp = 0;
                } else {
                    ck.reads[i_read].hp = 1;
                }
            }

            if constexpr (DEBUG_PRINT) {
                fprintf(stderr, "[dbg::%s] qn %s hp %d; cnt: %.1f %.1f %.1f\n", __func__,
                        ck.qnames[i_read].c_str(), ck.reads[i_read].hp, cnt[0], cnt[1], cnt[2]);
            }
        }
    }
}

std::vector<uint8_t> variant_graph_do_simple_haptag1(chunk_t &ck, const uint32_t seedreadID) {
    // Pick a read, haptag as hap0 and assign haptag 0
    // to its variants. Then for each iteration, haptag one read
    // with best score, and accumulate new phased variants
    // & discount known phased variants when there are conflicts.
    constexpr int DEBUG_PRINT = 0;
    variant_graph_t &vg = ck.vg;

    if (seedreadID >= ck.reads.size()) {  // should not happen
        return {};
    }

    std::vector<uint8_t> readhps(ck.reads.size(), HAPTAG_UNPHASED);
    std::vector<std::array<float, 3>> varhps(vg.n_vars, {0.0f, 0.0f, 0.0f});
    // counter of allele0-as-hp0, allele0-as-hp1 and the sum

    readhps[seedreadID] = 0;
    variant_graph_update_varhp_given_read(ck, varhps, seedreadID, 0);

    while (true) {
        uint32_t i_best = std::numeric_limits<uint32_t>::max();
        uint8_t hp_best = HAPTAG_UNPHASED;
        float score_best = 0;
        int updated_best = 0;

        // to the right of self read
        int n_consecutive_no_tagging = 0;
        for (size_t readID = seedreadID + 1; readID < ck.reads.size(); readID++) {
            if (ck.reads[readID].start_pos > ck.reads[seedreadID].end_pos) {
                break;
            }
            if (readhps[readID] != HAPTAG_UNPHASED) {
                continue;
            }
            infer_readhp_stat_t stat = variant_graph_infer_readhp_given_vars(
                    ck, varhps, static_cast<uint32_t>(readID));
            if (stat.updated_best < 1) {
                n_consecutive_no_tagging++;
                if (n_consecutive_no_tagging > 100) {
                    break;
                }
            } else {
                n_consecutive_no_tagging = 0;
                if (stat.score_best > score_best) {
                    score_best = stat.score_best;
                    updated_best = stat.updated_best;
                    hp_best = stat.hp_best;
                    i_best = static_cast<uint32_t>(readID);
                }
            }
        }

        // to the left of self read
        n_consecutive_no_tagging = 0;
        for (int readID = (int)seedreadID - 1; (int)readID >= 0; readID--) {
            if (ck.reads[readID].start_pos + 50000 <= ck.reads[seedreadID].start_pos) {
                break;
            }
            if (readhps[readID] != HAPTAG_UNPHASED) {
                continue;
            }
            infer_readhp_stat_t stat = variant_graph_infer_readhp_given_vars(
                    ck, varhps, static_cast<uint32_t>(readID));
            if (stat.updated_best < 1) {
                n_consecutive_no_tagging++;
                if (n_consecutive_no_tagging > 100) {
                    break;
                }
            } else {
                n_consecutive_no_tagging = 0;
                if (stat.score_best > score_best) {
                    score_best = stat.score_best;
                    updated_best = stat.updated_best;
                    hp_best = stat.hp_best;
                    i_best = readID;
                }
            }
        }

        if (hp_best != HAPTAG_UNPHASED) {
            readhps[i_best] = hp_best;
            variant_graph_update_varhp_given_read(ck, varhps, i_best, hp_best);
            if constexpr (DEBUG_PRINT) {
                fprintf(stderr, "[dbg::%s] updated qn %s (i=%d) as hp %d, score=%.5f, updated=%d\n",
                        __func__, ck.qnames[i_best].c_str(), (int)i_best, hp_best, score_best,
                        updated_best);
            }
        } else {
            break;
        }
    }
    return readhps;
}

std::unordered_map<uint32_t, uint8_t> variant_graph_do_simple_haptag1_give_ht(
        chunk_t &ck,
        const int seedreadID) {
    // (produce a hashtable of readID2haptag rather than an array of such info)

    constexpr int DEBUG_PRINT = 0;
    assert(seedreadID >= 0 && (uint32_t)seedreadID < ck.reads.size());
    if constexpr (DEBUG_PRINT) {
        fprintf(stderr, "[dbg::%s] start from qn %s\n", __func__, ck.qnames[seedreadID].c_str());
    }

    // counter of allele0-as-hp0, allele0-as-hp1 and the sum
    std::unordered_map<uint32_t, uint8_t> readID2hp;
    std::unordered_map<uint32_t, std::array<float, 3>> pos2counter;

    readID2hp[seedreadID] = 0;
    variant_graph_update_varhp_given_read_ht(ck, pos2counter, seedreadID, 0);
    if constexpr (DEBUG_PRINT) {
        for (auto &pair : pos2counter) {
            fprintf(stderr, "[dbg::%s] (init) pos %d : %.1f, %.1f\n", __func__, pair.first,
                    std::get<0>(pair.second), std::get<1>(pair.second));
        }
    }

    while (true) {
        uint32_t i_best = std::numeric_limits<uint32_t>::max();
        uint8_t hp_best = HAPTAG_UNPHASED;
        float score_best = 0;
        int updated_best = 0;
        int n_consecutive_no_tagging;

        // to the right of self read
        n_consecutive_no_tagging = 0;
        for (uint32_t readID = seedreadID + 1; readID < ck.reads.size(); readID++) {
            if (ck.reads[readID].start_pos > ck.reads[seedreadID].end_pos) {
                break;
            }
            if (readID2hp.find(readID) != readID2hp.end() && readID2hp[readID] != HAPTAG_UNPHASED) {
                continue;
            }
            infer_readhp_stat_t stat =
                    variant_graph_infer_readhp_given_vars_ht(ck, pos2counter, readID, (uint8_t)0);
            if (stat.updated_best < 1) {
                n_consecutive_no_tagging++;
                if (n_consecutive_no_tagging > 100) {
                    break;
                }
            } else {
                n_consecutive_no_tagging = 0;
                if (stat.score_best > score_best) {
                    score_best = stat.score_best;
                    updated_best = stat.updated_best;
                    hp_best = stat.hp_best;
                    i_best = readID;
                }
            }
        }

        // to the left of self read
        n_consecutive_no_tagging = 0;
        for (int readID = (int)seedreadID - 1; (int)readID >= 0; readID--) {
            if (ck.reads[readID].start_pos + 50000 <= ck.reads[seedreadID].start_pos) {
                break;
            }
            if (readID2hp.find(readID) != readID2hp.end() && readID2hp[readID] != HAPTAG_UNPHASED) {
                continue;
            }
            infer_readhp_stat_t stat =
                    variant_graph_infer_readhp_given_vars_ht(ck, pos2counter, readID, 0);
            if (stat.updated_best < 1) {
                n_consecutive_no_tagging++;
                if (n_consecutive_no_tagging > 100) {
                    break;
                }
            } else {
                n_consecutive_no_tagging = 0;
                if (stat.score_best > score_best) {
                    score_best = stat.score_best;
                    updated_best = stat.updated_best;
                    hp_best = stat.hp_best;
                    i_best = readID;
                }
            }
        }

        if (hp_best != HAPTAG_UNPHASED) {
            if constexpr (DEBUG_PRINT) {
                infer_readhp_stat_t stat =
                        variant_graph_infer_readhp_given_vars_ht(ck, pos2counter, i_best, 1);
                fprintf(stderr,
                        "[dbg::%s] updated qn %s (i=%d pos %d-%d) as hp %d, score=%.5f, "
                        "updated=%d (stat: %.2f, %d, %d)\n",
                        __func__, ck.qnames[i_best].c_str(), (int)i_best,
                        (int)ck.reads[i_best].start_pos, (int)ck.reads[i_best].end_pos, hp_best,
                        score_best, updated_best, stat.score_best, stat.updated_best, stat.hp_best);
            }
            readID2hp[i_best] = hp_best;
            variant_graph_update_varhp_given_read_ht(ck, pos2counter, i_best, hp_best);
        } else {
            break;
        }
    }

    return readID2hp;
}

}  // namespace kadayashi
