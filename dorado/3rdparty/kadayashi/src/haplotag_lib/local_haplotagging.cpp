#include "local_haplotagging.h"

#include "hts_types.h"
#include "types.h"

#include <htslib/bgzf.h>
#include <htslib/faidx.h>
#include <htslib/hts.h>
#include <htslib/sam.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace kadayashi {

#define MAX_READS 134217727  // note: need to modify pg_t to increase this
#define VAR_CALL_MIN_STRAND_PRESENCE 2
#define READ_MAX_DIVERG 0.1  // note: 0.05 would be too low
#define HAP_TAG_MAX_CNFLCT_EV 3
#define HAP_TAG_MAX_CNFLCT_RATIO 0.5
#define REFVAR_INDEX_BUCKET_L 10000

#define TRF_DEFAULT_K 5
#define TRF_MIN_TANDAM_DEPTH 1
#define TRF_MOTIF_MAX_LEN 200
#define TRF_ADD_PADDING 10
#define TRF_CLOSE_GAP_THRESHOLD 50

#define SENTINEL_REF_ALLELE "M"
#define SENTINEL_REF_ALLELE_L 1

constexpr bool DEBUG_LOCAL_HAPLOTAGGING = false;

namespace {

inline std::string create_region_string(const std::string &ref_name,
                                        const uint32_t start,
                                        const uint32_t end) {
    return ref_name + ":" + std::to_string(start) + "-" + std::to_string(end);
}

void vg_update_varhp_given_read(const chunk_t *ck,
                                std::vector<std::array<float, 3>> &varhps,
                                const int readID,
                                const uint8_t hp) {
    const read_t &r = ck->reads[readID];
    for (size_t i_var = 0; i_var < r.vars.size(); i_var++) {
        const uint32_t idx_var = r.vars[i_var].var_idx;
        if (r.vars[i_var].allele_idx > 1) {
            continue;
        }
        varhps[idx_var][hp ^ r.vars[i_var].allele_idx] += 1;
        varhps[idx_var][2] += 1;
    }
}

struct infer_readhp_t {
    float score_best;
    int updated_best;
    uint8_t hp_best;
};
infer_readhp_t vg_infer_readhp_given_vars(const chunk_t *ck,
                                          const std::vector<std::array<float, 3>> &varhps,
                                          const uint32_t readID) {
    const read_t &r = ck->reads[readID];
    float hp0 = 0.0;
    float hp1 = 0.0;
    int updated = 0;
    for (size_t i_var = 0; i_var < r.vars.size(); i_var++) {
        const uint32_t idx_var = r.vars[i_var].var_idx;
        const uint32_t idx_allele = r.vars[i_var].allele_idx;
        if (idx_allele > 1) {
            continue;
        }
        if (varhps[idx_var][2] < 0.1) {
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
        return hp0 > hp1 ? infer_readhp_t{.score_best = hp0, .updated_best = updated, .hp_best = 0}
                         : infer_readhp_t{.score_best = hp1, .updated_best = updated, .hp_best = 1};
    }
    return infer_readhp_t{.score_best = 0.0f, .updated_best = 0, .hp_best = HAPTAG_UNPHASED};
}

std::vector<uint8_t> vg_do_simple_haptag1(chunk_t *ck, const uint32_t seedreadID) {
    // Pick a read, haptag as hap0 and assign haptag 0
    // to its variants. Then for each iteration, haptag one read
    // with best score, and accumulate new phased variants
    // & discount known phased variants when there are conflicts.
    constexpr int DEBUG_PRINT = 0;
    vg_t &vg = ck->vg;

    if (seedreadID >= ck->reads.size()) {  // should not happen
        return {};
    }

    std::vector<uint8_t> readhps(ck->reads.size(), HAPTAG_UNPHASED);
    std::vector<std::array<float, 3>> varhps(vg.n_vars, {0.0f, 0.0f, 0.0f});
    // counter of allele0-as-hp0, allele0-as-hp1 and the sum

    readhps[seedreadID] = 0;
    vg_update_varhp_given_read(ck, varhps, seedreadID, 0);

    while (true) {
        uint32_t i_best = std::numeric_limits<uint32_t>::max();
        uint8_t hp_best = HAPTAG_UNPHASED;
        float score_best = 0;
        int updated_best = 0;
        for (size_t readID = 0; readID < ck->reads.size(); readID++) {
            if (readhps[readID] != HAPTAG_UNPHASED) {
                continue;
            }
            const infer_readhp_t stat =
                    vg_infer_readhp_given_vars(ck, varhps, static_cast<uint32_t>(readID));
            if (stat.score_best > score_best && stat.updated_best > 0) {
                score_best = stat.score_best;
                updated_best = stat.updated_best;
                hp_best = stat.hp_best;
                i_best = static_cast<uint32_t>(readID);
            }
        }
        if (hp_best != HAPTAG_UNPHASED) {
            readhps[i_best] = hp_best;
            vg_update_varhp_given_read(ck, varhps, i_best, hp_best);
            if constexpr (DEBUG_PRINT) {
                fprintf(stderr, "[dbg::%s] updated qn %s (i=%d) as hp %d, score=%.5f, updated=%d\n",
                        __func__, ck->qnames[i_best].c_str(), (int)i_best, hp_best, score_best,
                        updated_best);
            }
        } else {
            break;
        }
    }
    return readhps;
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
            int comparable = normalize_readtaggings_count(data[j], data[i]);
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

int sort_qa_v(std::vector<qa_t> &h) {
    if (h.size() <= 1) {
        return 0;
    }
    std::stable_sort(h.begin(), h.end(),
                     [](const qa_t &a, const qa_t &b) { return a.pos < b.pos; });
    return 1;
}

/*** bam parsing helpers ***/
std::unique_ptr<bamfile_t> init_bamfile_t_with_opened_files(samFile *fp_bam,
                                                            hts_idx_t *fp_bai,
                                                            sam_hdr_t *fp_header) {
    if (!fp_bam || !fp_bai || !fp_header) {
        return nullptr;
    }
    std::unique_ptr<bamfile_t> h = std::make_unique<bamfile_t>();
    h->fp = fp_bam;
    h->bai = fp_bai;
    h->header = fp_header;
    return h;
}

}  // namespace

// arrays
uint32_t max_of_u32_array(const uint32_t *a, const int l, int *idx) {
    uint32_t ret = 0;
    for (int i = 0; i < l; i++) {
        if (a[i] >= ret) {
            ret = a[i];
            if (idx) {
                *idx = i;
            }
        }
    }
    return ret;
}

// clang-format off
const unsigned char md_op_table[256]={
    // [0-9] gives 0
    // [^] gives 1
    // [ATCGatcgUuNn] gives 2
    // else: 4
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 4, 4,  4, 4, 4, 4,
    4, 2, 4, 2,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 2/*N*/, 4,
    4, 4, 4, 4,  2, 2, 4, 4,  4, 4, 4, 4,  4, 4, 1, 4,
    4, 2, 4, 2,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 2/*n*/, 4,
    4, 4, 4, 4,  2, 2, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
    4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
};
// clang-format on

static int sancheck_MD_tag_exists_and_is_valid(const bam1_t *aln) {
    const uint8_t *tmp = bam_aux_get(aln, "MD");
    if (!tmp) {
        return 0;
    }
    const char *md_s = bam_aux2Z(tmp);
    int i = 1;
    while (md_s[i]) {
        const int md_type = md_op_table[(int)md_s[i]];
        if (md_type >= 4) {
            return 0;
        }
        ++i;
    }
    return 1;
}

int natoi(const char *s, const int l) {
    // Convert trusted non-null terminated string to positive int.
    // Use only for parsing MD tag.
    // Return:
    //    -1 if error (invalid length, non-numeric character, overflow)
    //    INT upon successful conversion
    if (l <= 0 || l > 10) {
        return -1;
    }
    int ret = 0;
    int x, y;
    for (int i = 0; i < l; i++) {
        if (!s[i]) {
            return -1;
        }
        x = (int)s[i];
        if (x < 48 || x > 57) {
            return -1;
        }
        int e = 1;
        for (int j = 0; j < l - 1 - i; j++) {
            e *= 10;
        }
        if (i == 10 && x > 2) {
            return -1;  // overflow
        }
        y = (x - 48) * e;
        if (INT32_MAX - y < ret) {
            return -1;  // overflow
        }
        ret += y;
    }
    return ret;
}

static void seq2nt4seq(const char *seq, const int seq_l, std::vector<uint8_t> &h) {
    h.resize(seq_l);
    for (int i = 0; i < seq_l; i++) {
        h[i] = static_cast<uint8_t>(kdy_seq_nt4_table[static_cast<int>(seq[i])]);
    }
}

inline std::string nt4seq2seq(std::vector<uint8_t> &h, const int l) {
    std::string ret(l, '\0');
    for (int i = 0; i < l; i++) {
        ret[i] = "ACGT"[h[i]];
    }
    return ret;
}

static int add_allele_ta_t(ta_t &h,
                           const std::vector<uint8_t> &allele_nt4seq,
                           const uint32_t readID,
                           const int not_sure_is_new) {
    // (do nothing if allele is already in the buffer)
    // return: 1 if added; 0 if done nothing.
    if (not_sure_is_new) {
        for (size_t i = 0; i < h.alleles.size(); i++) {
            if (h.alleles[i] == allele_nt4seq) {
                // just log the readID
                h.allele2readIDs[i].push_back(readID);
                return 0;
            }
        }
    }

    // unseen allele, make slot and push the allele
    h.alleles.push_back(allele_nt4seq);

    // push readID
    h.allele2readIDs.push_back({readID});

    // push empty is-used flag
    h.alleles_is_used.push_back(0);

    return 1;
}

static void push_allele_ta_v(std::vector<ta_t> &h,
                             const uint32_t pos,
                             const std::vector<uint8_t> &allele_nt4seq,
                             const uint32_t readID) {
    // Note: assumes that we load variants in sorted order.
    // Here we will not search for `pos`, but instead just
    // check if the last position is `pos` (add allele)
    // or not (add position and add allele).
    if (!h.empty() && (h.back().pos == pos)) {
        const int i = static_cast<int>(h.size()) - 1;
        add_allele_ta_t(h[i], allele_nt4seq, readID, 1);
    } else {
        // new position
        h.push_back(ta_t{});
        h.back().pos = pos;
        add_allele_ta_t(h.back(), allele_nt4seq, readID, 0);
    }
}

static void cleanup_alleles_ta_t(ta_t &var) {
    size_t ih = 0;                                        // i_head
    for (size_t it = 0; it < var.alleles.size(); it++) {  // i_tail
        if (var.alleles_is_used[it]) {
            if (ih < it) {
                var.alleles_is_used[ih] = 1;
                var.allele2readIDs[ih] = var.allele2readIDs[it];
                var.alleles[ih] = var.alleles[it];
            }
            ih++;
        }
    }
    var.alleles.resize(ih);
    var.allele2readIDs.resize(ih);
    var.alleles_is_used.resize(ih);
}

static void add_allele_qa_v_nt4seq(std::vector<qa_t> &h,
                                   const uint32_t pos,
                                   const std::vector<uint8_t> &allele,
                                   const int allele_l,
                                   const uint8_t cigar_op) {
    h.push_back(qa_t{});
    h.back().pos = pos;
    h.back().is_used = 0;
    h.back().allele_idx = std::numeric_limits<uint32_t>::max();
    for (int i = 0; i < allele_l; i++) {
        h.back().allele.push_back(allele[i]);
    }
    // append cigar operation to the allele integer sequence
    h.back().allele.push_back(cigar_op);
}

static void add_allele_qa_v(std::vector<qa_t> &h,
                            const uint32_t pos,
                            const char *allele,
                            const int allele_l,
                            const uint8_t cigar_op) {
    h.push_back(qa_t{});
    h.back().pos = pos;
    h.back().is_used = 0;
    h.back().allele_idx = std::numeric_limits<uint32_t>::max();
    seq2nt4seq(allele, allele_l, h.back().allele);

    // append cigar operation to the allele integer sequence
    h.back().allele.push_back(cigar_op);
}

static void filter_lift_qa_v_given_conf_list(const std::vector<qa_t> &src,
                                             std::vector<qa_t> &dst,
                                             const ref_vars_t *refvars) {
    // assumes variants in src are sorted by position.

    if (refvars->poss.empty() || src.empty()) {
        return;
    }

    int start_idx = -1;
    for (size_t i = 0; i < src.size(); i++) {
        start_idx = refvars->start_indices[src[i].pos / REFVAR_INDEX_BUCKET_L];
        if (start_idx >= 0) {
            break;
        }
    }
    if (start_idx < 0) {
        return;
    }
    assert(start_idx >= 0 && ((uint32_t)start_idx) < refvars->poss.size());

    int i = 0;
    const int n_poss = static_cast<int>(refvars->poss.size()) - start_idx;
    while (i < n_poss - 1 && src[0].pos > refvars->poss[i + start_idx]) {
        i++;
    }
    for (size_t ir = 0; ir < src.size(); ir++) {
        uint32_t pos = src[ir].pos;
        if (pos < refvars->poss[i + start_idx]) {
            continue;
        } else if (pos >= refvars->poss[i + start_idx]) {
            while (i < n_poss - 1 && pos > refvars->poss[i + start_idx]) {
                i++;
            }
            if (pos == refvars->poss[i + start_idx]) {
                const qa_t &h = src[ir];
                // push
                add_allele_qa_v(dst, h.pos, SENTINEL_REF_ALLELE, SENTINEL_REF_ALLELE_L, VAR_OP_X);
                // copy over the actual allele sequence
                dst.back().allele.resize(h.allele.size());
                for (size_t j = 0; j < h.allele.size(); j++) {
                    dst.back().allele[j] = h.allele[j];
                }
                // shift ref var idx
                if (i < n_poss - 1) {
                    i++;
                }
            }
        }
    }
}

static int get_allele_index(const ta_t *h, const std::vector<uint8_t> &nt4seq) {
    for (int i = 0; i < static_cast<int>(h->alleles.size()); i++) {
        if (h->alleles[i] == nt4seq) {
            return i;
        }
    }
    return -1;
}

static inline unsigned char filter_base_by_qv(const char raw, const int min_qv) {
    return (int)raw - 33 >= min_qv ? raw : 'N';
}

static int parse_variants_for_one_read(const bam1_t *aln,
                                       std::vector<qa_t> &vars,
                                       const int min_base_qv,
                                       uint32_t *left_clip_len,
                                       uint32_t *right_clip_len) {
    // note: caller ensure that MD tag exists and
    //       does not have unexpected operations.
    // Return: 0 if ok, 1 when error
    const int SNPonly = 0;
    constexpr int DEBUG_PRINT = 0;
    int failed = 0;

    int self_start = 0;
    const uint32_t ref_start = static_cast<uint32_t>(aln->core.pos);

    // parse cigar for insertions
    std::vector<uint64_t> insertions;  // for parsing MD tag
    const uint32_t *cigar = bam_get_cigar(aln);
    const uint8_t *seqi = bam_get_seq(aln);
    uint32_t op, op_l;
    uint32_t ref_pos = ref_start;
    uint32_t self_pos = 0;
    if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
        fprintf(stderr, "[dbg::%s] at read %s (ref start pos=%u)\n", __func__, bam_get_qname(aln),
                ref_start);
    }
    for (uint32_t i = 0; i < aln->core.n_cigar; i++) {
        op = bam_cigar_op(cigar[i]);
        op_l = bam_cigar_oplen(cigar[i]);
        if (op == BAM_CREF_SKIP) {
            ref_pos += op_l;
        } else if (op == BAM_CSOFT_CLIP) {
            if (i == 0) {
                self_start = op_l;
                if (left_clip_len) {
                    *left_clip_len = op_l;
                }
            } else {
                if (right_clip_len) {
                    *right_clip_len = op_l;
                }
            }
            self_pos += op_l;
        } else if (op == BAM_CMATCH || op == BAM_CEQUAL || op == BAM_CDIFF) {
            ref_pos += op_l;
            self_pos += op_l;
        } else if (op == BAM_CINS) {
            std::string seq(op_l, '\0');
            for (uint32_t j = 0; j < op_l; j++) {
                seq[j] = filter_base_by_qv(seq_nt16_str[bam_seqi(seqi, self_pos + j)], min_base_qv);
            }
            if (!SNPonly) {
                add_allele_qa_v(vars, ref_pos, seq.c_str(), op_l, VAR_OP_I);  // push_to_vvar_t()
            }
            insertions.push_back(((uint64_t)op_l) << 32 | self_pos);
            self_pos += op_l;
        } else if (op == BAM_CDEL) {
            ref_pos += op_l;
        }
    }

    // parse MD tag for SNPs and deletions
    char snp_base[2];
    snp_base[1] = 0;
    char snp_base_dbg[10];
    snp_base_dbg[9] = 0;
    size_t prev_ins_idx = 0;
    const uint8_t *tagd = bam_aux_get(aln, "MD");
    const char *md_s = bam_aux2Z(tagd);
    int prev_md_i = 0;
    int prev_md_type, md_type;
    if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
        fprintf(stderr, "[dbg::%s] qn=%s\n", __func__, bam_get_qname(aln));
        fprintf(stderr, "[dbg::%s] MD=%s\n", __func__, md_s);
    }
    // (init)
    self_pos = self_start;
    ref_pos = ref_start;
    prev_md_type = md_op_table[(int)md_s[0]];
    if (prev_md_type == 2) {
        snp_base[0] = filter_base_by_qv(seq_nt16_str[bam_seqi(seqi, self_pos)], min_base_qv);
        add_allele_qa_v(vars, ref_pos, snp_base, 1, VAR_OP_X);
        ref_pos++;
        self_pos++;
        prev_md_type = -1;
    }
    if (prev_md_type >= 4) {
        failed = 1;
    }

    // (collect operations)
    int i = 1;
    while (!failed && md_s[i]) {
        md_type = md_op_table[(int)md_s[i]];
        if (md_type != prev_md_type) {
            if (prev_md_type == 0) {  // prev was match
                int l = natoi(md_s + prev_md_i, i - prev_md_i);
                if (l < 0) {
                    failed = 1;
                    break;
                }
                ref_pos += l;
                self_pos += l;
                while (prev_ins_idx < insertions.size() &&
                       self_pos > (uint32_t)insertions[prev_ins_idx]) {
                    self_pos += insertions[prev_ins_idx] >> 32;
                    prev_ins_idx++;
                }
            } else if (prev_md_type == 1) {  // prev was del
                if (md_type == 0) {          // current sees numeric, del run has ended
                    if (!SNPonly) {
                        add_allele_qa_v(vars, ref_pos, md_s + prev_md_i + 1, i - prev_md_i - 1,
                                        VAR_OP_D);
                    }
                    ref_pos += i - prev_md_i - 1;
                    prev_md_type = md_type;
                    prev_md_i = i;
                } else {  // still in a del run, do not update status
                    ;
                }
                i++;
                continue;
            }
            // is current a SNP?
            if (md_type == 2) {
                snp_base[0] =
                        filter_base_by_qv(seq_nt16_str[bam_seqi(seqi, self_pos)], min_base_qv);
                add_allele_qa_v(vars, ref_pos, snp_base, 1, VAR_OP_X);
                if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
                    for (uint32_t x = (self_pos > 7 ? self_pos - 7 : 0), y = 0; x < self_pos + 2;
                         x++, y++) {
                        snp_base_dbg[y] =
                                filter_base_by_qv(seq_nt16_str[bam_seqi(seqi, x)], min_base_qv);
                    }
                    fprintf(stderr,
                            "[dbg::%s] pushed SNP ref_pos=%d self_pos=%d base=%s, -7~+1:%s\n",
                            __func__, ref_pos, self_pos, snp_base, snp_base_dbg);
                }
                ref_pos++;
                self_pos++;
                prev_md_type = -1;
                prev_md_i = i;
            } else {
                prev_md_type = md_type;
                prev_md_i = i;
            }
        }
        i++;
    }

    if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
        for (size_t tmpi = 0; tmpi < vars.size(); tmpi++) {
            int tmpop = vars[tmpi].allele[vars[tmpi].allele.size() - 2];
            fprintf(stderr, "[dbg::%s]    op=%c pos=%d len=%d ", __func__, "MXID"[tmpop],
                    vars[tmpi].pos, (int)std::ssize(vars[tmpi].allele) - 1);
            if (tmpop > 0) {
                fprintf(stderr, "seq=");
                for (int64_t tmpj = 0; tmpj < std::ssize(vars[tmpi].allele) - 1; tmpj++) {
                    fprintf(stderr, "%c", "ACGT?"[vars[tmpi].allele[tmpj]]);
                }
            }
            fprintf(stderr, "\n");
        }
    }

    return failed;
}

struct pg_t {
    // LIMITATION: number of reads is at most (1<<27)-1
    uint64_t key;    // pos:32 | cigar:4 | strand:1 | readID:27
    uint32_t varID;  // as in the read
};  // piggyback
#define pg_t_cmp(x, y) ((x).key < (y).key)
static void push_to_pg_t(std::vector<pg_t> &pg,
                         const uint32_t i_read,
                         const uint32_t pos,
                         const uint8_t op,
                         const uint32_t i_var) {
    if (i_read >= (1UL << 27)) {
        fprintf(stderr, "[E::%s] i_read value does not fit into 27 bits. iread = %u\n", __func__,
                i_read);
        return;
    }
    uint64_t key = ((uint64_t)pos) << 32 | ((uint64_t)op) << 28 | ((uint64_t)op) << 27 | i_read;
    auto &last = pg.emplace_back(pg_t{});
    last.key = key;
    last.varID = i_var;
}

static int sort_qa_v_for_all(chunk_t *ck) {
    int tot = 0;
    for (size_t i = 0; i < ck->reads.size(); i++) {
        tot += sort_qa_v(ck->reads[i].vars);
    }
    return tot;
}

static int get_cigar_op_qa_t(const qa_t &h) {
    if (h.allele.empty()) {
        return VAR_OP_INVALID;
    }
    return h.allele.back();
}

static void log_allele_indices_for_reads_given_varcalls(chunk_t *ck) {
    // After pileup/unphased varcall, collect indices of
    // alleles of interest on each read (stored sorted wrt to variant's
    // reference positions), so that we will not need
    // to refer to actual allele sequences when constructing
    // the variant graph.

    sort_qa_v_for_all(ck);

    std::vector<ta_t> &vars = ck->varcalls;
    std::vector<qa_t> new_;
    for (size_t i_read = 0; i_read < ck->reads.size(); i_read++) {
        new_.clear();
        read_t &read = ck->reads[i_read];
        size_t prev_i = 0;
        for (uint32_t i_pos = 0; i_pos < vars.size(); i_pos++) {
            const uint32_t pos = vars[i_pos].pos;
            const ta_t *var = &vars[i_pos];
            if (!var->is_used) {
                continue;
            }
            if (pos < read.start_pos || pos >= read.end_pos) {
                continue;
            }
            for (size_t i = prev_i; i < read.vars.size(); i++) {
                const qa_t &readvar = read.vars[i];
                if (readvar.pos > pos) {
                    break;
                }
                if (readvar.pos == pos) {
                    const int allele_idx = get_allele_index(var, readvar.allele);
                    if (allele_idx < 0 || !var->alleles_is_used[allele_idx]) {
                        continue;
                    }

                    // if we reach here, current position and self allele is in use
                    add_allele_qa_v_nt4seq(
                            new_, pos, readvar.allele,
                            static_cast<int>(readvar.allele.size()) - 1,  // last slot is cigar op
                            readvar.allele[readvar.allele.size() - 1]);

                    // update flags and reverse index entries
                    new_.back().is_used = 1;
                    new_.back().hp = HAPTAG_UNPHASED;
                    new_.back().var_idx = i_pos;
                    new_.back().allele_idx = allele_idx;

                    // step forward
                    prev_i = i + 1;
                    break;
                }
            }
        }
        // replace with sorted, indexed entries
        read.vars.clear();
        for (size_t i = 0; i < new_.size(); i++) {
            auto &tmp = read.vars.emplace_back(new_[i]);
            tmp.is_used = 1;
        }
    }
}

static std::vector<uint64_t> TRF_heuristic(const char *seq, const int seq_l, const int ref_start) {
    // A simple tandem repeat masker inspired by TRF.
    // Uses arbitrary thresholds and does not exclude
    // non-tandem direct repeats, unlike the TRF.
    // The usecase is to merely avoid picking up variants
    // too close to low-complexity runs for phasing.
    constexpr int DEBUG_PRINT = 0;
    if (DEBUG_LOCAL_HAPLOTAGGING) {
        fprintf(stderr, "[dbg::%s] seq_l is %d, offset %d\n", __func__, seq_l, ref_start);
    }
    FILE *fp = 0;
    if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
        fp = fopen("test.bed", "w");
        assert(fp);
    }

    // length for exact match lenth
    int k = 5;  // note:  we will enumerate all combos

    // input sancheck
    if (seq_l < k * 3) {  // sequence too short, nothing to do
        return {};
    }

    // init buffer for kmer index
    uint32_t idx_l = 1;
    for (int i = 0; i < k; i++) {
        idx_l *= 4;
    }
    std::vector<std::vector<uint32_t>> idx(idx_l);
    std::vector<uint8_t> idx_good(idx_l, 0);
    if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
        fprintf(stderr, "[dbg::%s] idx_l %d\n", __func__, (int)idx_l);
    }

    // index kmers
    uint32_t mer = 0;
    const uint32_t mermask = (1 << (k * 2)) - 1;
    int mer_n = 0;
    for (int i = 0; i < seq_l; i++) {
        int c = (int)kdy_seq_nt4_table[(uint8_t)seq[i]];
        if (c != 4) {
            mer = (mer << 2 | c) & mermask;
            if (mer_n < k) {
                mer_n++;
            }
            if (mer_n == k) {
                idx[mer].push_back(i + ref_start - k + 1);  // use absolute coordinate
            }
        } else {
            mer = 0;
            mer_n = 0;
        }
    }
    if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
        for (uint32_t i = 0; i < idx_l; i++) {
            if (idx[i].size() != 0) {
                fprintf(stderr, "[dbg::%s] combo#%d ", __func__, i);
                for (int j = 0; j < k; j++) {
                    fprintf(stderr, "%c", "ACGT"[(i >> ((k - j - 1) * 2)) & 3]);
                }
                fprintf(stderr, " n=%d\n", (int)idx[i].size());
                if constexpr (DEBUG_PRINT > 1) {
                    fprintf(stderr, "   ^");
                    for (size_t j = 0; j < idx[i].size(); j++) {
                        fprintf(stderr, " %d", idx[i][j]);
                    }
                    fprintf(stderr, "\n");
                }
            }
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
        if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr, "[dbg::%s] ds is empty (ds0 was %d)\n", __func__, (int)ds0.size());
            for (size_t i = 0; i < ds0.size(); i++) {
                fprintf(stderr, "[dbg::%s] ds0 #%d is %d\n", __func__, (int)i, (int)ds0[i]);
            }
        }

    } else {
        if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr, "[dbg::%s] there are %d candidate distances (raw: %d)\n", __func__,
                    (int)ds.size(), (int)ds0.size());
            for (size_t i = 0; i < ds.size(); i++) {
                fprintf(stderr, "[dbg::%s] ds #%d is %d\n", __func__, (int)i, (int)ds[i]);
            }
        }

        // process the kmer chains
        std::vector<uint32_t> poss;
        std::vector<uint32_t> intervals_buf;
        std::vector<uint8_t> used(idx_l);
        int buf[3] = {0, 0, 0};
        int failed = 0;
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
                            poss.push_back((uint32_t)i);
                        }
                    } else if (i + 2 == std::ssize(idx[i_mer]) - 1) {
                        if (dists[i_mer][i + 1] == d || dists[i_mer][i + 2] == d) {
                            poss.push_back((uint32_t)i);
                            ok = 1;
                        } else {
                            ok = 0;
                        }
                    } else if (i + 1 == std::ssize(idx[i_mer]) - 1) {
                        if (dists[i_mer][i + 1] == d) {
                            poss.push_back((uint32_t)i);
                            poss.push_back((uint32_t)i + 1);
                            ok = 1;
                            stop = 1;
                        } else if (dists[i_mer][i] == d) {
                            poss.push_back((uint32_t)i);
                            ok = 1;
                            stop = 1;
                        } else {
                            ok = 0;
                            stop = 1;
                        }
                    } else {
                        if (DEBUG_LOCAL_HAPLOTAGGING) {
                            fprintf(stderr, "[E::%s] impossible\n", __func__);
                        }
                        failed = 1;
                        break;
                    }
                    if ((!ok) || stop) {
                        while (!poss.empty() && dists[i_mer][poss.back()] != d) {
                            poss.pop_back();
                        }
                        if (!poss.empty()) {
                            int start = idx[i_mer][poss[0]];
                            int end = idx[i_mer][poss.empty() ? 0 : poss.back()];
                            if (end - start > k + 2) {
                                if (start > TRF_ADD_PADDING + 1) {
                                    start -= k + TRF_ADD_PADDING + 1;
                                }
                                end += 2 * k + TRF_ADD_PADDING;
                                intervals_buf.push_back(start << 1);
                                intervals_buf.push_back(end << 1 | 1);
                                if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
                                    fprintf(stderr, "[dbg::%s] d=%d mer=%d saw start-end: %d-%d\n",
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
            if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
                fprintf(stderr,
                        "[dbg::%s] failed parsing mers (%d) or intervals_buf is empty (%d)\n",
                        __func__, failed, (int)intervals_buf.size());
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

                if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
                    fprintf(stderr, "[dbg::%s] %d stat=%d depth=%d lbreak=%d\n", __func__,
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
                                if (start - (uint32_t)intervals.back() <
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
                        if (start - (uint32_t)intervals.back() <
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
            if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
                fprintf(stderr, "[dbg::%s] had %d intervals\n", __func__, (int)intervals.size());
                for (size_t i = 0; i < intervals.size(); i++) {
                    fprintf(stderr, "[dbg::%s]    #%d %d-%d\n", __func__, (int)i,
                            (int)(intervals[i] >> 32), (int)(uint32_t)intervals[i]);
                }
            }

            if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
                for (size_t i = 0; i < intervals.size(); i++) {
                    fprintf(fp, "chr20\t%d\t%d\n", (int)(intervals[i] >> 32),
                            (int)(uint32_t)intervals[i]);
                }
            }
        }
    }

    if constexpr (DEBUG_PRINT) {
        fclose(fp);
    }
    return intervals;
}

static std::vector<uint64_t> get_lowcmp_mask(const faidx_t *fai,
                                             const char *ref_name,
                                             const int ref_start,
                                             const int ref_end) {
    const std::string span_str = create_region_string(ref_name, ref_start + 1, ref_end);

    char *refseq_s = 0;
    int refseq_l = -1;
    refseq_s = fai_fetch(fai, span_str.c_str(), &refseq_l);

    if (refseq_l <= 0 || !refseq_s) {
        if (DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr,
                    "[W::%s] failed to fetch reference sequence (name=%s, start=%d, end=%d; err "
                    "code %d)\n",
                    __func__, ref_name, ref_start, ref_end, refseq_l);
        }
        if (refseq_s) {
            hts_free(refseq_s);
        }
        return {};
    }
    std::vector<uint64_t> ret = TRF_heuristic(refseq_s, refseq_l, ref_start);
    if (refseq_s) {
        hts_free(refseq_s);
    }
    return ret;
}

chunk_t unphased_varcall_a_chunk(bamfile_t *hf,
                                 ref_vars_t *refvars,
                                 const faidx_t *ref_faidx,
                                 const char *refname,
                                 const int32_t itvl_start,
                                 const int32_t itvl_end,
                                 const pileup_pars_t &pp,
                                 const int disable_lowcmp_filter) {
    const int SNPonly = 1;
    constexpr int DEBUG_PRINT = 0;
    int failed = 0;

    const int min_mapq = 10;
    const float max_aln_de = static_cast<float>(READ_MAX_DIVERG);
    const float min_het_ratio = 0.1f;  // e.g. 0.1 means only take positions with
                                       // non-ref allele frequencies summed to [0.1, 0.9)
                                       // as possibly heterozygous.

    const int base_q_min = pp.allow_any_candidate ? 0 : pp.min_base_quality;
    const int32_t var_call_min_cov =
            pp.allow_any_candidate ? 0 : std::max(0, pp.min_varcall_coverage);
    const float var_call_min_cov_ratio = pp.allow_any_candidate ? 0 : pp.min_varcall_fraction;
    const int var_call_min_strand_presence =
            pp.allow_any_candidate ? 0 : VAR_CALL_MIN_STRAND_PRESENCE;
    const int32_t var_call_neigh_indel_flank =
            pp.allow_any_candidate ? 0 : std::max(0, pp.varcall_indel_mask_flank);

    assert(itvl_start >= 0);
    assert(itvl_end >= 0);
    chunk_t ck = {
            .is_valid = 1,
            .reads = {},
            .varcalls = {},
            .qnames = {},
            .qname2ID = {},
            .abs_start = static_cast<uint32_t>(itvl_start),
            .abs_end = static_cast<uint32_t>(itvl_end),
            .vg = {},
    };

    const std::string itvl = create_region_string(refname, itvl_start, itvl_end);
    HtsItrPtr bamitr =
            HtsItrPtr(sam_itr_querys(hf->bai, hf->header, itvl.c_str()), HtsItrDestructor());
    BamPtr aln = BamPtr(bam_init1(), BamDestructor());

    // Adjust region start and end: if there happens to be no
    // heterozygous variants around start and/or end, we could have
    // 30-50% unread unphased. Here, we allow expanding the
    // target region *somewhat*: the start/end of the first/last
    // read, if they are not too long; or +-50kb when they are
    // too long.
    uint32_t abs_start = ck.abs_start;
    uint32_t abs_end = ck.abs_end;
    int offset_default = 50000;
    if (!pp.disable_region_expansion) {
        int offset_left = offset_default;
        int offset_right = 0;

        // left
        const std::string itvl_left = create_region_string(refname, itvl_start, (itvl_start + 1));
        bamitr = HtsItrPtr(sam_itr_querys(hf->bai, hf->header, itvl_left.c_str()),
                           HtsItrDestructor());
        while (sam_itr_next(hf->fp, bamitr.get(), aln.get()) >= 0) {
            if (itvl_start - aln->core.pos < offset_default) {
                offset_left = static_cast<int>(itvl_start - aln->core.pos);
                break;
            }
        }

        //right
        const std::string itvl_right = create_region_string(refname, itvl_end, (itvl_end + 1));
        bamitr = HtsItrPtr(sam_itr_querys(hf->bai, hf->header, itvl_right.c_str()),
                           HtsItrDestructor());
        while (sam_itr_next(hf->fp, bamitr.get(), aln.get()) >= 0) {
            int end_pos = (int)bam_endpos(aln.get());
            if (end_pos - itvl_end < offset_default) {
                offset_right =
                        offset_right > end_pos - itvl_end ? offset_right : end_pos - itvl_end;
            }
        }
        if (offset_right == 0) {
            offset_right = offset_default;
        }

        // update itr
        abs_start = itvl_start - offset_left < 0 ? 0 : itvl_start - offset_left;
        abs_end = itvl_end + offset_right;
        const std::string itvl_abs = create_region_string(refname, abs_start, abs_end);
        bamitr = HtsItrPtr(sam_itr_querys(hf->bai, hf->header, itvl_abs.c_str()),
                           HtsItrDestructor());
        if (DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr, "[M::%s] interval expansion: now using %s (-%d, +%d)\n", __func__,
                    itvl_abs.c_str(), offset_left, offset_right);
        }
    }

    uint32_t n_reads = 0;
    std::vector<qa_t> tmp_qav;
    while (sam_itr_next(hf->fp, bamitr.get(), aln.get()) >= 0) {
        n_reads++;
        if (n_reads > MAX_READS) {
            if (DEBUG_LOCAL_HAPLOTAGGING) {
                fprintf(stderr, "[E::%s] too many reads\n", __func__);
            }
            failed = 1;
            break;
        }
        const std::string qn(bam_get_qname(aln.get()));
        const int flag = aln->core.flag;
        const int mapq = (int)aln->core.qual;
        const uint8_t *tmp = bam_aux_get(aln.get(), "de");
        float de = 0;
        if (tmp) {
            de = static_cast<float>(bam_aux2f(tmp));
        }

        // require MD tag to exist and does not have unexpected operations
        const int md_is_ok = sancheck_MD_tag_exists_and_is_valid(aln.get());
        if (!md_is_ok) {
            continue;
        }

        if (aln->core.n_cigar == 0) {
            continue;
        }
        if ((flag & 4) || (flag & 256) || (flag & 2048)) {
            continue;
        }
        if (mapq < min_mapq) {
            continue;
        }
        if (de > max_aln_de) {
            continue;
        }

        // collect variants of the read
        read_t new_read{
                .start_pos = static_cast<uint32_t>(aln->core.pos),
                .end_pos = static_cast<uint32_t>(bam_endpos(aln.get())),
                .ID = static_cast<uint32_t>(std::size(ck.reads)),
                .strand = !!(flag & 16),
                .de = de,
        };

        if (!refvars) {
            const int parse_failed =
                    parse_variants_for_one_read(aln.get(), new_read.vars, base_q_min,
                                                &new_read.left_clip_len, &new_read.right_clip_len);
            if (!parse_failed) {
                sort_qa_v(new_read.vars);

                // read info
                if (ck.qname2ID.find(qn) != ck.qname2ID.end()) {
                    if (DEBUG_LOCAL_HAPLOTAGGING) {
                        fprintf(stderr, "[W::%s] dup read name? qn=%s\n", __func__, qn.c_str());
                    }
                } else {
                    ck.reads.emplace_back(std::move(new_read));
                    ck.qnames.emplace_back(qn);
                    ck.qname2ID[qn] = new_read.ID;
                }
            }
        } else {
            tmp_qav.clear();
            const int parse_failed =
                    parse_variants_for_one_read(aln.get(), tmp_qav, base_q_min,
                                                &new_read.left_clip_len, &new_read.right_clip_len);
            if (!parse_failed) {
                sort_qa_v(tmp_qav);
                filter_lift_qa_v_given_conf_list(tmp_qav, new_read.vars, refvars);

                // read info
                if (ck.qname2ID.find(qn) != ck.qname2ID.end()) {
                    if (DEBUG_LOCAL_HAPLOTAGGING) {
                        fprintf(stderr, "[W::%s] dup read name? qn=%s\n", __func__, qn.c_str());
                    }
                } else {
                    ck.reads.emplace_back(std::move(new_read));
                    ck.qnames.emplace_back(qn);
                    ck.qname2ID[qn] = new_read.ID;
                }
            }
        }
        if (ck.reads.size() > pp.max_read_per_region) {  // the user-imposed limit
            failed = 1;
            break;
        }
    }
    aln = {};
    bamitr = {};

    if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
        fprintf(stderr, "[dbg::%s] chunk has %d reads\n", __func__, (int)ck.reads.size());
    }

    if (ck.reads.size() < 2 || failed) {
        ck.is_valid = 0;
        return ck;
    }

    // collect hp/lowcmp mask
    std::vector<uint64_t> hplowcmp_mask;
    if (!disable_lowcmp_filter) {
        hplowcmp_mask = get_lowcmp_mask(ref_faidx, refname, abs_start, abs_end);
    }

    // piggyback sorting pileup 1st pass: without ref calls
    const uint64_t mask_readID = ~(std::numeric_limits<uint64_t>::max() << 27);
    std::vector<pg_t> pg;
    for (uint32_t i_read = 0; i_read < ck.reads.size(); i_read++) {
        std::vector<qa_t> &vars = ck.reads[i_read].vars;
        for (uint32_t i_pos = 0; i_pos < vars.size(); i_pos++) {
            const qa_t &var = vars[i_pos];

            if (var.pos < abs_start || var.pos >= abs_end) {
                continue;
            }
            if (SNPonly && get_cigar_op_qa_t(var) != VAR_OP_X) {
                continue;
            }
            int is_banned = 0;
            if (!hplowcmp_mask.empty()) {
                for (size_t i = 0; i < hplowcmp_mask.size(); i++) {
                    if ((var.pos >= (hplowcmp_mask[i] >> 32)) &&
                        (var.pos < static_cast<uint32_t>(hplowcmp_mask[i]))) {
                        is_banned = 1;
                        break;
                    }
                }
            }
            if (is_banned) {
                continue;
            }

            // don't store the variant if allele has N
            int is_sus = 0;
            if (var.allele.size() >= 1) {
                for (int64_t i = 0; i < std::ssize(var.allele) - 1; i++) {
                    if (var.allele[i] > 3) {
                        is_sus = 1;
                        break;
                    }
                }
            }
            if (is_sus) {
                continue;
            }

            if (SNPonly) {  // TODO: better heuristics for remove
                            //       false variants around homopolymers.
                // Here, we simply check if there's any indel residing too close
                // to the current SNP. If so, just discard the SNP.
                if (i_pos > 0) {
                    const qa_t &var_l = vars[i_pos - 1];
                    if (var_l.allele.empty()) {  // (should not happen)
                        continue;
                    }
                    const uint8_t var_l_op = static_cast<uint8_t>(get_cigar_op_qa_t(var_l));
                    if ((var.pos <= (var_call_neigh_indel_flank + var_l.pos)) &&
                        (var_l_op == VAR_OP_I)) {
                        continue;
                    }
                    if (var_l_op == VAR_OP_D) {  // (for del, use end position of the del run
                                                 // rather than start position.)
                        const uint32_t cigar_l = static_cast<uint32_t>(
                                var_l.allele.size() -
                                1);  // var_l.allele is guaranteed to have more than 1 entry
                        if (var.pos <= (var_call_neigh_indel_flank + var_l.pos + cigar_l)) {
                            continue;
                        }
                    }
                }
                if (i_pos + 1 < vars.size()) {
                    const qa_t &var_r = vars[i_pos + 1];
                    if ((var_r.pos <= (var_call_neigh_indel_flank + var.pos)) &&
                        (get_cigar_op_qa_t(var_r) & (VAR_OP_D | VAR_OP_I))) {
                        continue;
                    }
                }
            }

            push_to_pg_t(pg, i_read, var.pos, static_cast<uint8_t>(get_cigar_op_qa_t(var)), i_pos);
        }
    }

    if (pg.empty()) {
        if (DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr, "[M::%s] no variant to call (%s:%d-%d)\n", __func__, refname,
                    ck.abs_start, ck.abs_end);
        }
        ck.is_valid = 0;
        return ck;
    }
    std::sort(pg.begin(), pg.end(), [](const pg_t &a, const pg_t &b) { return a.key < b.key; });

    // prepare the 2nd piggyback sorting: add in ref calls
    // for positions that may be het.
    // If we collect ref later it would a pita having to have
    // an index sentinel. Better to have an allele sequence sentinel instead.
    int has_ref_allele = 1;
    int n_added_ref = 0;
    int strand_count[2];
    int strand;
    const int64_t pgn = static_cast<int64_t>(pg.size());
    for (int64_t i = 0; i + 1 < pgn; /******/) {
        const uint32_t pos = pg[i].key >> 32;
        strand_count[0] = 0;
        strand_count[1] = 0;
        strand = !!(ck.reads[pg[i].key & mask_readID].strand);
        strand_count[strand]++;

        // check if has enough alts and fair read strands
        int64_t j;
        for (j = i + 1; j < static_cast<int64_t>(pg.size()); j++) {
            if ((pg[j].key >> 32) != pos) {
                break;
            }
            strand = !!(ck.reads[pg[j].key & mask_readID].strand);
            strand_count[strand]++;
        }
        if (j - i < var_call_min_cov) {
            i = j;
            continue;
        }
        if (MIN(strand_count[0], strand_count[1]) < var_call_min_strand_presence) {
            i = j;
            continue;
        }

        // check if looking like hom
        int cov_tot = 0;
        for (size_t i_read = 0; i_read < ck.reads.size(); i_read++) {  // TODO: is this slow
            if (ck.reads[i_read].start_pos <= pos && ck.reads[i_read].end_pos > pos) {
                cov_tot++;
            }
        }
        const float call_ratio = (float)(j - i) / cov_tot;
        if (call_ratio < min_het_ratio || call_ratio >= (1 - min_het_ratio)) {
            i = j;
            continue;
        }
        // this position might be a het candidate, now collect
        // ref calls.
        for (size_t i_read = 0; i_read < ck.reads.size(); i_read++) {
            if (ck.reads[i_read].start_pos >= pos || ck.reads[i_read].end_pos < pos) {
                continue;
            }
            std::vector<qa_t> &readvars = ck.reads[i_read].vars;
            int has_alt = 0;
            for (size_t i_var = 0; i_var < readvars.size(); i_var++) {
                if (readvars[i_var].pos == pos) {
                    has_alt = 1;
                    break;
                }
                if (readvars[i_var].pos > pos) {
                    // look back and see if the position is actually covered by a DEL
                    // in which case we should not collect a ref allele.
                    if (i_var > 0) {
                        const uint32_t cigar_i =
                                static_cast<uint32_t>(readvars[i_var - 1].allele.size() -
                                                      1);  // size() is guaranteed to be >1
                        const uint8_t cigar_op = readvars[i_var - 1].allele[cigar_i];
                        if (cigar_op == VAR_OP_D) {
                            uint32_t cigar_l = cigar_i;
                            if (readvars[i_var - 1].pos + cigar_l >= pos) {
                                has_alt =
                                        2;  // not an alt starting from `pos`, but also not the ref allele.
                                break;
                            }
                        }
                    }
                    if (!has_alt) {
                        if (SNPonly) {
                            // check if any closeby indel
                            // TODO: better heurstics (see notes above about homopolymer)
                            qa_t &var = readvars[i_var];
                            if (i_var > 0) {
                                qa_t &var_l = readvars[i_var - 1];
                                if ((pos <= var_l.pos + var_call_neigh_indel_flank) &&
                                    (get_cigar_op_qa_t(var_l) & (VAR_OP_D | VAR_OP_I))) {
                                    has_alt = 3;
                                    break;
                                }
                            }
                            if ((var.pos <= pos + var_call_neigh_indel_flank) &&
                                (get_cigar_op_qa_t(var) & (VAR_OP_D | VAR_OP_I))) {
                                has_alt = 4;
                            }
                        }
                    }
                    break;
                }
            }
            if (!has_alt) {  // add ref allele to both the piggyback buffer
                             // and the read's collection of variants
                add_allele_qa_v(readvars, pos, SENTINEL_REF_ALLELE, SENTINEL_REF_ALLELE_L,
                                VAR_OP_X);
                push_to_pg_t(pg, static_cast<uint32_t>(i_read), pos, static_cast<uint8_t>(VAR_OP_X),
                             static_cast<uint32_t>(readvars.size() -
                                                   1));  // readvars guaranteed to be not empty here
                n_added_ref++;
            }
        }
        i = j;
    }
    if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
        fprintf(stderr, "[dbg::%s] added %d for 2nd piggyback sorting; pg is now length %d\n",
                __func__, n_added_ref, (int)pg.size());
    }

    // note: do not sort qa_v here , piggyback relies on allele as-in-buffer indexing

    // piggyback sorting pileup 2nd pass
    std::sort(pg.begin(), pg.end(), [](const pg_t &a, const pg_t &b) { return a.key < b.key; });

    // Finding het variants:
    //  1) drop if read varaints' cov<3, or one strand unseen
    //  2) collect total coverage, drop if looking like hom
    //  3) compare alleles, collect the solid ones if any
    for (size_t i = 0; i + 1 < pg.size(); /****/) {  // loop over positions
        const uint32_t pos = pg[i].key >> 32;

        size_t j = i + 1;
        for (j = i + 1; j < pg.size(); j++) {
            if ((pg[j].key >> 32) != pos) {
                break;
            }
        }
        const int cov_tot = static_cast<int>(j) - static_cast<int>(i);
        if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr, "[dbg::%s] pos=%d checkpoint0 ; cov(with ref allele)=%d\n", __func__,
                    (int)pos, cov_tot);
        }

        // collect allele sequences
        // Since in most locations the number of possible alleles should be
        // small, here we just linear search to dedup.
        int pushed_a_var = 0;
        for (size_t k = i; k < i + cov_tot; k++) {
            const uint64_t rID = pg[k].key & mask_readID;

            // for 2a-diploid case, do not count contributions from reads with large clippings
            if (ck.reads[rID].left_clip_len > (uint32_t)pp.max_clipping ||
                ck.reads[rID].right_clip_len > (uint32_t)pp.max_clipping) {
                if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
                    fprintf(stderr,
                            "[dbg::%s]  allelic cov skipped qn=%s (left clip %d, right %d)\n",
                            __func__, ck.qnames[rID].c_str(), ck.reads[rID].left_clip_len,
                            ck.reads[rID].right_clip_len);
                }
                continue;
            }
            pushed_a_var = 1;

            const uint32_t varID = pg[k].varID;
            const qa_t &read_var = ck.reads[rID].vars[varID];
            if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
                fprintf(stderr,
                        "[dbg::%s] collecting allelic coverage: qn=%s var_pos=%d var_len=%d (first "
                        "char %d)\n",
                        __func__, ck.qnames[rID].c_str(), read_var.pos,
                        (int)read_var.allele.size() - 1,
                        read_var.allele[0]);  // size() guaranteed to >1
            }
            push_allele_ta_v(ck.varcalls, pos, read_var.allele, static_cast<uint32_t>(rID));
        }
        if (!pushed_a_var || ck.varcalls.empty()) {
            i = j;  // step forward
            continue;
        }

        ta_t &var = ck.varcalls.back();

        // (check allele occurences, and N base occurence)
        if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr, "[dbg::%s] pos=%d checkpoint3 n_alleles=%d; var call min cov is %d\n",
                    __func__, pos, (int)var.alleles.size(), var_call_min_cov);
        }
        int n_cov_passed = 0;
        size_t threshold = var_call_min_cov;  // TODO: may need to modify this for multi-allele
        if (var_call_min_cov_ratio > 0) {
            int tmptot = 0;
            for (size_t k = 0; k < var.alleles.size(); k++) {
                tmptot += static_cast<int>(var.allele2readIDs[k].size());
            }
            uint32_t threshold2 = (uint32_t)(tmptot * var_call_min_cov_ratio);
            threshold = threshold > threshold2 ? threshold : threshold2;
        }

        for (size_t k = 0; k < var.alleles.size(); k++) {
            if (var.allele2readIDs[k].size() >= threshold) {
                var.alleles_is_used[k] = 1;
                n_cov_passed++;
            } else {
                var.alleles_is_used[k] = 0;
            }
        }
        if (n_cov_passed == (has_ref_allele ? 2 : 1)) {  // 2-allele diploid case
            var.is_used = 1;
            if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
                fprintf(stderr, "[dbg::%s] pos %d accept (sancheck: %d %d\n", __func__, pos,
                        (int)var.allele2readIDs[0].size(), (int)var.allele2readIDs[1].size());
            }
            // Drop any unused alleles now.
            // They are not useful after this point and
            // will complicate variant graph ds.
            cleanup_alleles_ta_t(var);

        } else {  // drop the position now and free up the slot
            var.is_used = 0;
            ck.varcalls.pop_back();
            if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
                fprintf(stderr, "[dbg::%s] pos %d reject\n", __func__, pos);
            }
        }

        // step forward
        i = j;
    }

    // if there are variants close the the start/end, do not
    // allow variants from outside of the requested interval.
    // selfnote: buffer no longer clean; it has been kept clean
    // until here.

    // clean up read variants using the called variants
    log_allele_indices_for_reads_given_varcalls(&ck);
    if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
        fprintf(stderr, "[dbg::%s] collected vars on read; varcall size: %d\n", __func__,
                static_cast<int>(ck.varcalls.size()));
    }

    return ck;
}

static void vg_get_edge_values1(const chunk_t *ck,
                                const uint32_t varID1,
                                const uint32_t varID2,
                                int counter[4]) {
    for (int i = 0; i < 4; i++) {
        counter[i] = 0;
    }
    assert(varID1 < varID2);
    for (const read_t &r : ck->reads) {
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

static int diff_of_top_two(uint32_t d[4]) {
    std::sort(&d[0], &d[0] + 4);
    return d[3] - d[2];
}

static int vg_pos_is_confident(const vg_t &vg, const int var_idx) {
    uint32_t tmpcounter[4] = {0, 0, 0, 0};
    for (int tmpi = 0; tmpi < 4; tmpi++) {
        tmpcounter[tmpi] = vg.nodes[var_idx].scores[tmpi];
    }
    int ret = diff_of_top_two(tmpcounter) > 5;
    return ret;
}

static int vg_init_scores_for_a_location(chunk_t *ck, const uint32_t var_idx, int do_wipeout) {
    // return 1 when re-init was a hard one
    constexpr int DEBUG_PRINT = 0;
    int ret = -1;
    vg_t &vg = ck->vg;
    const uint32_t pos = ck->varcalls[var_idx].pos;
    int counter[4] = {0, 0, 0, 0};

    if (var_idx != 0 && !do_wipeout) {
        // reuse values from a previous position
        int i = 0;
        int diff = 5;
        for (int k = var_idx - 1; k > 0; k--) {
            const uint32_t pos2 = ck->varcalls[k].pos;
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
                    uint32_t tmpcounter[4] = {0, 0, 0, 0};
                    for (int tmpi = 0; tmpi < 4; tmpi++) {
                        tmpcounter[tmpi] = vg.nodes[k].scores[tmpi];
                    }
                    const int tmpdiff = diff_of_top_two(tmpcounter);
                    if constexpr (DEBUG_PRINT) {
                        fprintf(stderr,
                                "trying pos_resume %d (k=%d) checkpoint 2; %d %d %d %d ; %d\n",
                                pos2, k, tmpcounter[0], tmpcounter[1], tmpcounter[2], tmpcounter[3],
                                tmpdiff);
                    }
                    if (tmpdiff > diff) {
                        diff = tmpdiff;
                        i = k;
                    }
                }
            }
        }
        if (i > 0 /* && ok*/) {
            if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
                fprintf(stderr, "[dbg::%s] *maybe* re-init pos %d using pos %d\n", __func__,
                        (int)pos, (int)ck->varcalls[i].pos);
                for (int tmpi = 0; tmpi < 4; tmpi++) {
                    fprintf(stderr, "[dbg::%s] original %d : %d\n", __func__, i,
                            GET_VGN_VAL2(vg, var_idx, tmpi));
                }
            }
            vg_get_edge_values1(ck, (uint32_t)i, (uint32_t)var_idx, counter);

            // we forced to grab at least one resume point,
            // need to check if there's any read supporting the connection.
            // if not, resort to hard init as this is probably a
            // real phasing break.
            if (counter[0] < 3 && counter[1] < 3 && counter[2] < 3 && counter[3] < 3) {
                do_wipeout = 1;
                // This should go directly to the "maybewipe" part.

            } else {
                if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
                    fprintf(stderr,
                            "[dbg::%s] *maybe* not wiping; edge counts are %d %d %d %d; var idx "
                            "are %d and %d\n",
                            __func__, counter[0], counter[1], counter[2], counter[3], i, var_idx);
                }

                uint32_t bests[4];
                vgnode_t *n1 = &vg.nodes[var_idx];
                for (int tmpi = 0; tmpi < 4; ++tmpi) {
                    n1->scores[tmpi] = vg.nodes[i].scores[tmpi];
                    bests[tmpi] = n1->scores[tmpi];
                }
                int best_i = 0;
                max_of_u32_array(bests, 4, &best_i);
                n1->best_score_i = best_i;

                if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
                    for (int tmpi = 0; tmpi < 4; tmpi++) {
                        fprintf(stderr, "[dbg::%s] new %d : %d\n", __func__, tmpi,
                                GET_VGN_VAL2(vg, var_idx, tmpi));
                    }
                }
            }
        } else {
            if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
                fprintf(stderr, "[dbg::%s] tried but failed to find resume point\n", __func__);
            }
            do_wipeout = 1;  // did not find a good resume point, will hard re-init
        }
    }

    // Previously: maybewipe section.
    if (var_idx == 0 || do_wipeout) {
        ret = 1;
        if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr, "[dbg::%s] hard re-init for pos %d\n", __func__, (int)pos);
        }
        for (int i = 0; i < 4; i++) {
            counter[i] = 0;
        }
        for (const read_t &r : ck->reads) {
            if (r.start_pos > pos) {
                break;
            }  // reads are loaded from sorted bam, safe to break here
            if (r.end_pos <= pos) {
                continue;
            }
            for (const qa_t &var : r.vars) {
                if (var.var_idx == var_idx) {
                    if (var.allele_idx == 0) {
                        counter[0]++;
                    } else if (var.allele_idx == 1) {
                        counter[3]++;
                    }
                    break;
                }
            }
        }
        GET_VGN_VAL2(vg, var_idx, 0) = counter[0];
        GET_VGN_VAL2(vg, var_idx, 1) = counter[0] + counter[3];
        GET_VGN_VAL2(vg, var_idx, 2) = 0;
        GET_VGN_VAL2(vg, var_idx, 3) = counter[3];
    } else {
        ret = 0;
    }

    int best_i = 0;
    uint32_t tmp[4];
    for (int i = 0; i < 4; i++) {
        tmp[i] = GET_VGN_VAL2(vg, var_idx, i);
    }
    max_of_u32_array(tmp, 4, &best_i);
    vg.nodes[var_idx].best_score_i = best_i;
    for (int i = 0; i < 4; i++) {
        vg.nodes[var_idx].scores_source[i] = 4;  // sentinel
    }
    assert(ret >= 0);
    return ret;
}

/*** 2-allele diploid local variant graph ***/
bool vg_gen(chunk_t *ck) {
    if (!ck || !ck->is_valid || std::empty(ck->varcalls)) {
        return false;
    }

    vg_t &vg = ck->vg;

    vg.n_vars = static_cast<uint32_t>(std::size(ck->varcalls));
    vg.nodes.resize(vg.n_vars);
    vg.edges.resize(vg.n_vars - 1);
    vg.next_link_is_broken.resize(vg.n_vars);

    // fill in nodes
    for (size_t i = 0; i < ck->varcalls.size(); i++) {
        assert(ck->varcalls[i].is_used);
        vg.nodes[i] = {.ID = static_cast<uint32_t>(i)};
        vg.nodes[i].del = 0;
    }

    // fill in edges
    for (int64_t i_read = 0; i_read < std::ssize(ck->reads); i_read++) {
        const read_t &r = ck->reads[i_read];
        if (!r.vars.empty()) {
            for (int64_t i = 0; i < std::ssize(r.vars) - 1; i++) {
                assert(r.vars[i].is_used == 1);
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
    vg_init_scores_for_a_location(ck, 0, 1);

    return true;
}

static void vg_propogate_one_step(chunk_t *ck, int *i_prev_, const int i_self) {
    // note: we redo initialization at cov dropouts rather than
    // let backtracing figure out these phasing breakpoints.
    // The breakpoints are stored in vg. When haptagging
    // a read, we will not mix evidences from different phase blocks.
    constexpr int DEBUG_PRINT = 0;
    assert(i_self > 0);

    vg_t &vg = ck->vg;
    int i_prev = *i_prev_;
    vgnode_t *n1 = &vg.nodes[i_self];

    uint32_t bests[4];
    if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
        const std::string a0 =
                nt4seq2seq(ck->varcalls[i_self].alleles[0],
                           static_cast<int>(ck->varcalls[i_self].alleles[0].size()) - 1);
        const std::string a1 =
                nt4seq2seq(ck->varcalls[i_self].alleles[1],
                           static_cast<int>(ck->varcalls[i_self].alleles[1].size()) - 1);
        fprintf(stderr, "[dbg::%s] i_self=%d (pos=%d a1=%s a2=%s):\n", __func__, i_self,
                ck->varcalls[i_self].pos, a0.c_str(), a1.c_str());
    }

    // check whether we have a coverage dropout
    for (int i = 0; i < 4; i++) {
        bests[i] = GET_VGE_VAL2(vg, i_prev, i);
    }
    const uint32_t best = max_of_u32_array(bests, 4, 0);
    if (best < 3) {  // less than 3 reads support any combination, spot is
                     // a coverage dropout, redo initialization.
        int reinit_failed = vg_init_scores_for_a_location(ck, i_self, 0);
        if (reinit_failed) {
            vg.next_link_is_broken[i_prev] = 1;
            vg.has_breakpoints = 1;
        }
        if (!vg_pos_is_confident(vg, i_self)) {
            vg.next_link_is_broken[i_self] = 1;
        }
        *i_prev_ = i_self;
        if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr, "[dbg::%s]    ! phasing broke at %d (coverage dropout)\n", __func__,
                    ck->varcalls[i_self].pos);
        }
    }

    for (uint8_t self_combo = 0; self_combo < 4; self_combo++) {
        uint32_t score[4];
        for (uint8_t prev_combo = 0; prev_combo < 4; prev_combo++) {
            int both_hom =
                    ((self_combo == 0 || self_combo == 3) && (prev_combo == 0 || prev_combo == 3));
            int s1 = GET_VGN_VAL2(vg, i_prev, prev_combo);
            int s2 = GET_VGE_VAL(vg, i_prev, prev_combo >> 1, self_combo >> 1);
            int s3 = both_hom ? 0 : GET_VGE_VAL(vg, i_prev, prev_combo & 1, self_combo & 1);

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
        bests[self_combo] = max_of_u32_array(score, 4, &source);
        n1->scores[self_combo] = bests[self_combo];
        n1->scores_source[self_combo] = static_cast<uint8_t>(source);
    }
    int best_i = 0;
    max_of_u32_array(bests, 4, &best_i);
    n1->best_score_i = best_i;

    // another check: if phasing is broken, redo init for self
    if (best_i == 0 || best_i == 3) {  // decision was hom
        int reinit_failed = vg_init_scores_for_a_location(ck, i_self, 0);
        if (reinit_failed) {
            vg.next_link_is_broken[i_prev] = 1;
            vg.has_breakpoints = 1;
        }
        if (!vg_pos_is_confident(vg, i_self)) {
            vg.next_link_is_broken[i_self] = 1;
        }
        if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr, "[dbg::%s]    ! phasing broke at %d (hom decision = %d)\n", __func__,
                    ck->varcalls[i_self].pos, best_i);
        }
    }

    *i_prev_ = i_self;
    if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
        fprintf(stderr, "[dbg::%s]    best i: %d (bests: %d %d %d %d)\n", __func__, best_i,
                bests[0], bests[1], bests[2], bests[3]);
    }
}
void vg_propogate(chunk_t *ck) {
    // Fill in scores for nodes.
    int i_prev = 0;
    for (uint32_t i = 1; i < ck->vg.n_vars; i++) {
        vg_propogate_one_step(ck, &i_prev, i);
    }
}

void vg_haptag_reads(chunk_t *ck) {
    // 2-allele diploid, dvr method
    // Given phased variants, assign haptags to reads.
    constexpr int DEBUG_PRINT = 0;
    vg_t &vg = ck->vg;
    int sancheck_cnt[5] = {
            0, 0,
            0,  // no variant
            0,  // ambiguous
            0,  // unphasd due to conflict
    };
    uint32_t var_i_start = 0;
    uint32_t var_i_end = vg.n_vars;
    std::vector<uint64_t> buf;

    for (size_t i_read = 0; i_read < ck->reads.size(); i_read++) {
        read_t *r = &ck->reads[i_read];
        if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr, "[dbg::%s] saw qn %s\n", __func__, ck->qnames[i_read].c_str());
        }

        if (r->vars.empty()) {
            if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
                fprintf(stderr, "[dbg::%s] skip %s (no var)\n", __func__,
                        ck->qnames[i_read].c_str());
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
                            ck->qnames[i_read].c_str());
                }
                r->hp = HAPTAG_UNPHASED;
                sancheck_cnt[2]++;
                continue;
            }

            // (get largest block)
            std::sort(buf.begin(), buf.end());
            var_i_start = (uint32_t)buf.back();
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
                    fprintf(stderr, "%d, ", ck->varcalls[i].pos);
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
                uint32_t pos = ck->varcalls[r->vars[i].var_idx].pos;
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
                            ck->qnames[i_read].c_str(), r->vars[i_pos].pos, idx, combo);
                }
                votes[0]++;
            } else if (idx == (combo & 1)) {
                if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
                    fprintf(stderr, "[dbg::%s]    %s pos=%d hap 1 (idx=%d combo=%d)\n", __func__,
                            ck->qnames[i_read].c_str(), r->vars[i_pos].pos, idx, combo);
                }
                votes[1]++;
            } else {
                fprintf(stderr,
                        "[E::%s] %s qn=%d impossible (combo=%d idx=%d), check code. This read will "
                        "be untagged.\n",
                        __func__, ck->qnames[i_read].c_str(), r->vars[i_pos].pos, combo, idx);
                votes[0] = 0;
                votes[1] = 0;
                break;
            }
        }
        if (votes[0] > votes[1] && votes[0] > veto) {
            if (/*votes[1]<=HAP_TAG_MAX_CNFLCT_EV && */ (float)votes[1] / votes[0] <=
                HAP_TAG_MAX_CNFLCT_RATIO) {
                r->hp = 0;
                sancheck_cnt[0]++;
            } else {
                r->hp = HAPTAG_UNPHASED;
                sancheck_cnt[4]++;
            }
        } else if (votes[1] > votes[0] && votes[1] > veto) {
            if (/*votes[0]<=HAP_TAG_MAX_CNFLCT_EV && */ (float)votes[0] / votes[1] <=
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
                    ck->qnames[i_read].c_str(), votes[0], votes[1], veto, r->hp);
        }
    }
    if (DEBUG_LOCAL_HAPLOTAGGING) {
        fprintf(stderr,
                "[M::%s] n_reads %d, hap0=%d hap1=%d no_variant=%d ambiguous=%d "
                "unphased_due_conflict=%d\n",
                __func__, (int)ck->reads.size(), sancheck_cnt[0], sancheck_cnt[1], sancheck_cnt[2],
                sancheck_cnt[3], sancheck_cnt[4]);
    }
}

void vg_do_simple_haptag(chunk_t *ck, const uint32_t n_iter_requested) {
    constexpr int DEBUG_PRINT = 0;

    uint32_t n_iter = n_iter_requested;
    if (n_iter == 0 || n_iter >= ck->reads.size()) {
        n_iter = static_cast<int>(ck->reads.size());
    }
    const uint32_t stride = static_cast<int>(ck->reads.size()) / n_iter;

    std::vector<std::vector<uint8_t>> results;
    results.reserve(n_iter);

    for (uint32_t i_iter = 0; i_iter < n_iter; i_iter++) {
        // used the read with the most number of phasing variants within
        // the current bin
        uint32_t max_var = 0;
        uint32_t i_max_var = i_iter * stride;
        for (uint32_t j = i_iter * stride; j < (i_iter + 1) * stride; j++) {
            if (ck->reads[j].vars.size() > max_var) {
                max_var = static_cast<uint32_t>(ck->reads[j].vars.size());
                i_max_var = j;
            }
        }
        const uint32_t seedreadID = i_max_var;

        if constexpr (DEBUG_PRINT) {
            fprintf(stderr, "[dbg::%s] iter %d seed is %s (%d/%d, n_var=%d)\n", __func__, i_iter,
                    ck->qnames[i_iter].c_str(), (int)seedreadID, (int)ck->reads.size(),
                    (int)ck->reads[seedreadID].vars.size());
        }

        std::vector<uint8_t> readhps = vg_do_simple_haptag1(ck, seedreadID);
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
        for (uint32_t i_read = 0; i_read < ck->reads.size(); i_read++) {
            float cnt[3] = {0, 0, 0};
            for (const auto &result : results) {
                if (result[i_read] == HAPTAG_UNPHASED) {
                    cnt[2] += 1;
                } else {
                    cnt[result[i_read]] += 1;
                }
            }

            if ((cnt[0] > 3 && cnt[1] > 3) || (cnt[0] + cnt[1] < 0.5)
                //||((cnt[0] + cnt[1] > 0 && cnt[2] / (cnt[0] + cnt[1] + cnt[2]) > 0.5))
            ) {
                ck->reads[i_read].hp = HAPTAG_UNPHASED;
            } else {
                if (cnt[0] > cnt[1]) {
                    ck->reads[i_read].hp = 0;
                } else {
                    ck->reads[i_read].hp = 1;
                }
            }

            if constexpr (DEBUG_PRINT) {
                fprintf(stderr, "[dbg::%s] qn %s hp %d; cnt: %.1f %.1f %.1f\n", __func__,
                        ck->qnames[i_read].c_str(), ck->reads[i_read].hp, cnt[0], cnt[1], cnt[2]);
            }
        }
    }
}

phase_return_t kadayashi_local_haptagging_dvr_single_region1(samFile *fp_bam,
                                                             hts_idx_t *fp_bai,
                                                             sam_hdr_t *fp_header,
                                                             const faidx_t *fai,
                                                             const char *ref_name,
                                                             const uint32_t ref_start,
                                                             const uint32_t ref_end,
                                                             const pileup_pars_t &pp) {
    // Return:
    //    Unless hashtable's initialization or input opening failed,
    //    this function always return a hashtable, which might be empty
    //    if no read can be tagged or there was any error
    //    when tagging reads.
    phase_return_t ret;

    std::unique_ptr<bamfile_t> hf = init_bamfile_t_with_opened_files(fp_bam, fp_bai, fp_header);
    if (!hf) {
        return {};
    }

    chunk_t ck = unphased_varcall_a_chunk(hf.get(), 0, fai, ref_name, ref_start, ref_end, pp,
                                          0  // disable low complexity masking?
    );

    const bool vg_ok = vg_gen(&ck);
    if (vg_ok) {
        vg_propogate(&ck);
        vg_haptag_reads(&ck);
        for (size_t i = 0; i < ck.reads.size(); i++) {
            const int haptag = ck.reads[i].hp + 1;  // use 1-index
            ret.qname2hp[ck.qnames[i]] = haptag;
        }
    }
    ret.ck = std::move(ck);

    return ret;
}

phase_return_t kadayashi_local_haptagging_simple_single_region1(samFile *fp_bam,
                                                                hts_idx_t *fp_bai,
                                                                sam_hdr_t *fp_header,
                                                                const faidx_t *fai,
                                                                const char *ref_name,
                                                                const uint32_t ref_start,
                                                                const uint32_t ref_end,
                                                                const pileup_pars_t &pp) {
    phase_return_t ret;

    std::unique_ptr<bamfile_t> hf = init_bamfile_t_with_opened_files(fp_bam, fp_bai, fp_header);
    if (!hf) {
        return {};
    }

    chunk_t ck = unphased_varcall_a_chunk(hf.get(), 0, fai, ref_name, ref_start, ref_end, pp,
                                          1  // disable low complexity masking?
    );

    const bool vg_ok = vg_gen(&ck);
    if (vg_ok) {
        vg_do_simple_haptag(&ck, 20);
        for (size_t i = 0; i < ck.reads.size(); i++) {
            const int haptag = ck.reads[i].hp + 1;  // use 1-index
            ret.qname2hp[ck.qnames[i]] = haptag;
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
        const std::string &ref_name,
        const uint32_t ref_start,
        const uint32_t ref_end,
        const int disable_interval_expansion,
        const int min_base_quality,
        const int min_varcall_coverage,
        const float min_varcall_fraction,
        const int varcall_indel_mask_flank,
        const int max_clipping,
        const int max_read_per_region) {
    const pileup_pars_t pp = {
            .disable_region_expansion = static_cast<bool>(!!disable_interval_expansion),
            .allow_any_candidate = false,
            .min_base_quality = min_base_quality,
            .min_varcall_coverage = min_varcall_coverage,
            .min_varcall_fraction = min_varcall_fraction,
            .varcall_indel_mask_flank = varcall_indel_mask_flank,
            .max_clipping = max_clipping,
            .max_read_per_region = static_cast<uint64_t>(max_read_per_region),
    };

    phase_return_t result = kadayashi::kadayashi_local_haptagging_dvr_single_region1(
            fp_bam, fp_bai, fp_header, fai, ref_name.c_str(), ref_start, ref_end, pp);
    return result.qname2hp;
}

}  // namespace kadayashi
