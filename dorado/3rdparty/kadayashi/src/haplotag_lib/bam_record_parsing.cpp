#include "bam_record_parsing.h"

#include "sequence_utility.h"

#include <algorithm>
#include <cstdio>

namespace kadayashi {

namespace {

inline unsigned char filter_base_by_qv(char raw, int min_qv) {
    return static_cast<int>(raw) - 33 >= min_qv ? raw : 'N';
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

//static float gap_compressed_seq_identity_from_read_vars(const std::vector<qa_t> &readvars,
//                                                        int n_matches) {
//    // note/TODO:
//    //   n_matches must be supplied because we do not guarantee `readvars`
//    //   has everything (due to base Q filter, having required SNP-only, etc).
//    //   Maybe the cigar parsing step should be write to a buffer first and then
//    //   be filtered into the read set. That way it's also easier to
//    //    1) unify cigar+MD, full cigar and cs tag parsing. The latter two are not implemented yet.
//    //    2) allow this function to be used in case `de:f` isn't supplied.
//    float X = 0;
//    float inss = 0;
//    float dels = 0;
//    //float inss_len = 0;
//    //float dels_len = 0;
//    for (const qa_t &var : readvars) {
//        if (var.allele[0] == SENTINEL_REF_ALLELE_INT) {
//            continue;
//        }
//        if (var.allele.back() == VAR_OP_X) {
//            X++;
//        } else if (var.allele.back() == VAR_OP_D) {
//            dels++;
//            //dels_len += var.allele.size()-1;
//        } else if (var.allele.back() == VAR_OP_I) {
//            inss++;
//            //inss_len += var.allele.size()-1;
//        } else {  // should not happen
//            fprintf(stderr, "[E::%s] shouldn't happen\n", __func__);
//            ;
//        }
//    }
//    // note: mm2 also counts ambiguous bases in the denominator. See `mm_event_identity`.
//    float n_amb = 0;
//    float M = static_cast<float>(n_matches);
//    float de = 1.0f - M / (M + n_amb + X + inss + dels);
//    return de;
//}
}  // namespace

int sancheck_MD_tag_exists_and_is_valid(const bam1_t *aln) {
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

void add_allele_qa_v(std::vector<qa_t> &h,
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

int parse_variants_for_one_read(const bam1_t *aln,
                                std::vector<qa_t> &vars,
                                const int min_base_qv,
                                int *left_clip_len,
                                int *right_clip_len,
                                const int SNPonly,
                                BlockedBloomFilter *bf) {
    // note: caller ensure that MD tag exists and
    //       does not have unexpected operations.
    // note2: bloom filter:
    //        (1)if bloom filter is provided and is not set to be frozen,
    //        then when a position is seen for the first time,
    //        it will be inserted into the bloom filter and not collected
    //        into the read. This is only intended to be used in the
    //        initial unphased pileup, and caller should ensure no race.
    //        (2)If the bloom filter is provided and is frozen, we will check
    //        with it and only collect known variants.
    // Return: 0 if ok, 1 when error
    constexpr int DEBUG_PRINT = 0;
    int failed = 0;

    int self_start = 0;
    const uint32_t ref_start = static_cast<uint32_t>(aln->core.pos);

    // parse cigar for insertions
    std::vector<uint64_t> insertions;  // for parsing MD tag
    const uint32_t *cigar = bam_get_cigar(aln);
    const uint8_t *seqi = bam_get_seq(aln);
    uint32_t ref_pos = ref_start;
    uint32_t self_pos = 0;
    if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
        fprintf(stderr, "[dbg::%s] at read %s (ref start pos=%u)\n", __func__, bam_get_qname(aln),
                ref_start);
    }

    uint32_t op = std::numeric_limits<uint32_t>::max();
    uint32_t op_l = 0;
    for (uint32_t i = 0; i < aln->core.n_cigar; i++) {
        op = bam_cigar_op(cigar[i]);
        op_l = bam_cigar_oplen(cigar[i]);
        if (op == BAM_CREF_SKIP) {
            ref_pos += op_l;
        } else if (op == BAM_CSOFT_CLIP) {
            if (i == 0) {
                self_start = op_l;
                if (left_clip_len) {
                    *left_clip_len = static_cast<int>(op_l);
                }
            } else {
                if (right_clip_len) {
                    *right_clip_len = static_cast<int>(op_l);
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
                const bool do_insert = check_blockedbloomfilter_to_decide_inserting(bf, ref_pos);
                if (do_insert) {
                    add_allele_qa_v(vars, ref_pos, seq.c_str(), op_l,
                                    VAR_OP_I);  // push_to_vvar_t()
                }
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
    const std::string md_ss = md_s;
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
        const bool do_insert = check_blockedbloomfilter_to_decide_inserting(bf, ref_pos);
        if (do_insert) {
            add_allele_qa_v(vars, ref_pos, snp_base, 1, VAR_OP_X);
        }
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
                const int l =
                        std::stoi(md_ss.substr(prev_md_i, i));  // character at i is the operation
                if (l < 0) {
                    failed = 1;
                    break;
                }
                ref_pos += l;
                self_pos += l;
                while (prev_ins_idx < insertions.size() &&
                       self_pos >= static_cast<uint32_t>(insertions[prev_ins_idx])) {
                    self_pos += insertions[prev_ins_idx] >> 32;
                    prev_ins_idx++;
                }
            } else if (prev_md_type == 1) {  // prev was del
                if (md_type == 0) {          // current sees numeric, del run has ended
                    if (!SNPonly) {
                        const bool do_insert =
                                check_blockedbloomfilter_to_decide_inserting(bf, ref_pos);
                        if (do_insert) {
                            add_allele_qa_v(vars, ref_pos, md_s + prev_md_i + 1, i - prev_md_i - 1,
                                            VAR_OP_D);
                        }
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
                if (check_blockedbloomfilter_to_decide_inserting(bf, ref_pos)) {
                    add_allele_qa_v(vars, ref_pos, snp_base, 1, VAR_OP_X);
                }
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
    std::stable_sort(vars.begin(), vars.end(),
                     [](const qa_t &a, const qa_t &b) { return a.pos < b.pos; });

    if constexpr (DEBUG_PRINT && DEBUG_LOCAL_HAPLOTAGGING) {
        for (size_t tmpi = 0; tmpi < vars.size(); tmpi++) {
            const int tmpop = vars[tmpi].allele[vars[tmpi].allele.size() - 2];
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

}  // namespace kadayashi
