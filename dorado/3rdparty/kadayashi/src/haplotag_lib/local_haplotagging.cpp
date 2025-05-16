// clang-format off
#include "local_haplotagging.h"

#include "ksort.h"
#include "kvec.h"
#include "types.h"

#include <htslib/bgzf.h>
#include <htslib/faidx.h>
#include <htslib/hts.h>
#include <htslib/khash.h>
#include <htslib/khash_str2int.h>
#include <htslib/kstring.h>
#include <htslib/sam.h>

#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <memory>
#include <stdexcept>

// clang-format on

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

// arrays
uint32_t max_of_u32_array(const uint32_t *a, int l, int *idx) {
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
static int get_posint_digits(int n) {
    assert(n>=0);
    if (n<10)         {return 1;}
    if (n<100)        {return 2;}
    if (n<1000)       {return 3;}
    if (n<10000)      {return 4;}
    if (n<100000)     {return 5;}
    if (n<1000000)    {return 6;}
    if (n<10000000)   {return 7;}
    if (n<100000000)  {return 8;}
    if (n<1000000000) {return 9;}
    return 10;
}
// clang-format on

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

static int sancheck_MD_tag_exists_and_is_valid(bam1_t *aln) {
    uint8_t *tmp = bam_aux_get(aln, "MD");
    if (!tmp) {
        return 0;
    }
    char *md_s = bam_aux2Z(tmp);
    int i = 1;
    int md_type;
    while (md_s[i]) {
        md_type = md_op_table[(int)md_s[i]];
        if (md_type >= 4) {
            return 0;
        }
        ++i;
    }
    return 1;
}

int natoi(const char *s, int l) {
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

static int nt4seq_cmp(const vu8_t *s1, const vu8_t *s2) {
    // return: 0 if equal, 1 otherwise
    if (s1->n != s2->n) {
        return 1;
    }
    for (size_t i = 0; i < s1->n; i++) {
        if (s1->a[i] != s2->a[i]) {
            return 1;
        }
    }
    return 0;
}

static void seq2nt4seq(const char *seq, int seq_l, vu8_t *h) {
    h->n = 0;
    for (int i = 0; i < seq_l; i++) {
        kv_push(uint8_t, *h, (uint8_t)kdy_seq_nt4_table[(int)seq[i]]);
    }
}

static vchar_t *nt4seq2seq(vu8_t *h, int l) {
    vchar_t *ret = (vchar_t *)calloc(1, sizeof(vchar_t));
    assert(ret);
    for (int i = 0; i < l; i++) {
        kv_push(char, *ret, "ACGT"[h->a[i]]);
    }
    kv_push(char, *ret, 0);
    return ret;
}

/*** bam parsing helpers ***/
bamfile_t *init_bamfile_t_with_opened_files(samFile *fp_bam,
                                            hts_idx_t *fp_bai,
                                            sam_hdr_t *fp_header) {
    if (!fp_bam || !fp_bai || !fp_header) {
        return NULL;
    }
    bamfile_t *h = (bamfile_t *)calloc(1, sizeof(bamfile_t));
    if (!h) {
        if (DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr, "[E::%s] calloc failed\n", __func__);
        }
        return NULL;
    }
    h->fp = fp_bam;
    h->bai = fp_bai;
    h->header = fp_header;
    h->aln = bam_init1();
    return h;
}
void destroy_holder_bamfile_t(bamfile_t *h, int include_self) {
    bam_destroy1(h->aln);
    if (include_self) {
        free(h);
    }
}

static ta_v *init_ta_v() {
    ta_v *h = (ta_v *)calloc(1, sizeof(ta_v));
    assert(h);
    return h;
}
static int add_allele_ta_t(ta_t *h,
                           const vu8_t *allele_nt4seq,
                           uint32_t readID,
                           int not_sure_is_new) {
    // (do nothing if allele is already in the buffer)
    // return: 1 if added; 0 if done nothing.
    if (not_sure_is_new) {
        for (size_t i = 0; i < h->alleles.n; i++) {
            if (nt4seq_cmp(&h->alleles.a[i], allele_nt4seq) == 0) {
                // just log the readID
                kv_push(uint32_t, h->allele2readIDs.a[i], readID);
                return 0;
            }
        }
    }

    // unseen allele, alloc and push the allele
    vu8_t tmp;
    kv_init(tmp);
    for (size_t i = 0; i < allele_nt4seq->n; i++) {
        kv_push(uint8_t, tmp, allele_nt4seq->a[i]);
    }
    kv_push(vu8_t, h->alleles, tmp);

    // alloc and push readID
    vu32_t tmpnames;
    kv_init(tmpnames);
    kv_push(uint32_t, tmpnames, readID);
    kv_push(vu32_t, h->allele2readIDs, tmpnames);

    // push empty is-used flag
    kv_push(uint8_t, h->alleles_is_used, 0);

    return 1;
}

static void push_allele_ta_v(ta_v *h, uint32_t pos, const vu8_t *allele_nt4seq, uint32_t readID) {
    // Note: assumes that we load variants in sorted order.
    // Here we will not search for `pos`, but instead just
    // check if the last position is `pos` (add allele)
    // or not (add position and add allele).
    if (h->n > 0 && h->a[h->n - 1].pos == pos) {
        const int i = static_cast<int>(h->n) - 1;
        add_allele_ta_t(&h->a[i], allele_nt4seq, readID, 1);
    } else {
        // new position, need alloc
        ta_t tmp;
        kv_init(tmp.alleles);
        kv_init(tmp.allele2readIDs);
        kv_init(tmp.alleles_is_used);
        tmp.pos = pos;
        add_allele_ta_t(&tmp, allele_nt4seq, readID, 0);
        kv_push(ta_t, *h, tmp);
    }
}

static void cleanup_alleles_ta_t(ta_t *var) {
    size_t ih = 0;                                    // i_head
    for (size_t it = 0; it < var->alleles.n; it++) {  // i_tail
        if (var->alleles_is_used.a[it]) {
            if (ih < it) {
                var->alleles_is_used.a[ih] = 1;

                vu32_t *h1_src = &var->allele2readIDs.a[it];
                vu32_t *h1_dest = &var->allele2readIDs.a[ih];
                h1_dest->n = 0;
                for (size_t i = 0; i < h1_src->n; i++) {
                    kv_push(uint32_t, *h1_dest, h1_src->a[i]);
                }

                vu8_t *h2_src = &var->alleles.a[it];
                vu8_t *h2_dest = &var->alleles.a[ih];
                h2_dest->n = 0;
                for (size_t i = 0; i < h2_src->n; i++) {
                    kv_push(uint8_t, *h2_dest, h2_src->a[i]);
                }
            }
            ih++;
        }
    }
    for (size_t i = ih; i < var->alleles.n; i++) {
        kv_destroy(var->alleles.a[i]);
        kv_destroy(var->allele2readIDs.a[i]);
    }
    var->alleles.n = ih;
    var->allele2readIDs.n = ih;
    var->alleles_is_used.n = ih;
}

static void add_allele_qa_v_nt4seq(qa_v *h,
                                   uint32_t pos,
                                   const vu8_t *allele,
                                   int allele_l,
                                   uint8_t cigar_op) {
    qa_t tmp;
    tmp.pos = pos;
    tmp.is_used = 0;
    tmp.allele_idx = UINT32_MAX;
    kv_init(tmp.allele);
    for (int i = 0; i < allele_l; i++) {
        kv_push(uint8_t, tmp.allele, allele->a[i]);
    }
    // append cigar operation to the allele integer sequence
    kv_push(uint8_t, tmp.allele, cigar_op);

    kv_push(qa_t, *h, tmp);
}

static void add_allele_qa_v(qa_v *h,
                            uint32_t pos,
                            const char *allele,
                            int allele_l,
                            uint8_t cigar_op) {
    qa_t tmp;
    tmp.pos = pos;
    tmp.is_used = 0;
    tmp.allele_idx = UINT32_MAX;
    kv_init(tmp.allele);
    seq2nt4seq(allele, allele_l, &tmp.allele);

    // append cigar operation to the allele integer sequence
    kv_push(uint8_t, tmp.allele, cigar_op);

    kv_push(qa_t, *h, tmp);
}

static void wipe_qa_v(qa_v *h) {
    for (size_t i = 0; i < h->n; i++) {
        kv_destroy(h->a[i].allele);
    }
    h->n = 0;
}

static void filter_lift_qa_v_given_conf_list(const qa_v *src, qa_v *dst, ref_vars_t *refvars) {
    // assumes variants in src are sorted by position.

    if (refvars->poss.n == 0 || src->n == 0) {
        return;
    }

    uint32_t *poss = 0;  // just points to a slice, no allocation
    int start_idx = -1;
    for (size_t i = 0; i < src->n; i++) {
        start_idx = refvars->start_indices.a[src->a[i].pos / REFVAR_INDEX_BUCKET_L];
        if (start_idx >= 0) {
            poss = refvars->poss.a + start_idx;
            break;
        }
    }
    if (!poss) {
        return;
    }
    assert(start_idx >= 0 && ((uint32_t)start_idx) < refvars->poss.n);

    int i = 0;
    const int n_poss = static_cast<int>(refvars->poss.n) - start_idx;
    while (i < n_poss - 1 && src->a[0].pos > poss[i]) {
        i++;
    }
    for (size_t ir = 0; ir < src->n; ir++) {
        uint32_t pos = src->a[ir].pos;
        if (pos < poss[i]) {
            continue;
        } else if (pos >= poss[i]) {
            while (i < n_poss - 1 && pos > poss[i]) {
                i++;
            }
            if (pos == poss[i]) {
                qa_t *h = &src->a[ir];
                // push
                add_allele_qa_v(dst, h->pos, SENTINEL_REF_ALLELE, SENTINEL_REF_ALLELE_L, VAR_OP_X);
                // copy over the actual allele sequence
                kv_resize(uint8_t, dst->a[dst->n - 1].allele, h->allele.n);
                for (size_t j = 0; j < h->allele.n; j++) {
                    dst->a[dst->n - 1].allele.a[j] = h->allele.a[j];
                }
                // shift ref var idx
                if (i < n_poss - 1) {
                    i++;
                }
            }
        }
    }
}

static int get_allele_index(const ta_t *h, const vu8_t *nt4seq) {
    for (int i = 0; i < static_cast<int>(h->alleles.n); i++) {
        if (nt4seq_cmp(&h->alleles.a[i], nt4seq) == 0) {
            return i;
        }
    }
    return -1;
}

static int check_if_two_reads_are_compatible(const read_t *r1, const read_t *r2, int verbose) {
    // Assumes the variant buffers are clean, and has ref calls,
    // i.e. you must have already done the unphased varcall.
    // Check SNPs of the reads to see if the two collections are identical in
    // the reference range that they overlap.
    // Return
    //   1 if compatible
    //   0 if incompatible
    //   -1 if no variant available (apart, or homo region or one of the read
    //                               is too short/unfortunately placed)
    //   -2 if no or too few phasing evidences
    int debug_print = 0 || verbose;

    const int max_incompat =
            MAX(2, static_cast<int>(MIN(r1->vars->n, r2->vars->n)) / 40);  // TODO this is not great

    if (r1->vars->n == 0 || r2->vars->n == 0) {
        return -1;
    }

    size_t i1 = 0;
    size_t i2 = 0;
    // move to the shared start position
    if (r1->vars->a[0].pos < r2->vars->a[0].pos) {
        i1++;
        while (i1 < r1->vars->n) {
            if (r1->vars->a[i1].pos == r2->vars->a[0].pos) {
                break;
            } else if (r1->vars->a[i1].pos > r2->vars->a[0].pos) {
                break;
            }
            i1++;
        }
    } else if (r2->vars->a[0].pos < r1->vars->a[0].pos) {
        i2++;
        while (i2 < r2->vars->n) {
            if (r1->vars->a[0].pos == r2->vars->a[i2].pos) {
                break;
            } else if (r1->vars->a[0].pos < r2->vars->a[i2].pos) {
                break;
            }
            i2++;
        }
    }

    // check alleles
    int failed = 0;
    int n_match = 0;
    while (i1 < r1->vars->n && i2 < r2->vars->n) {
        if (r1->vars->a[i1].pos < r2->vars->a[i2].pos) {
            i1++;
            continue;
        }
        if (r2->vars->a[i2].pos < r1->vars->a[i1].pos) {
            i2++;
            continue;
        }

        if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr, "[dbg::%s]   check pos %d\n", __func__, r1->vars->a[i1].pos);
        }
        if (r1->vars->a[i1].allele_idx != r2->vars->a[i2].allele_idx) {
            failed++;
            if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
                fprintf(stderr, "[dbg::%s]   ^ FAILED (allowance at %d/%d)\n", __func__, failed,
                        max_incompat);
            }
        }
        if (failed > max_incompat) {
            break;
        }
        i1++;
        i2++;
        n_match++;
    }

    if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
        fprintf(stderr, "[dbg::%s] ^-- %d incompat, %d tot\n", __func__, failed, n_match);
    }
    if (failed <= max_incompat) {
        assert(i1 == r1->vars->n || i2 == r2->vars->n);
    } else {
        return 0;  // incompat
    }

    if (n_match - failed > 0) {
        return 1;  //  compat
    } else {
        return -2;  // not enough evidence
    }
}

static chunk_t *init_chunk_t(uint32_t start, uint32_t end) {
    int init_size = 128;
    chunk_t *h = (chunk_t *)calloc(1, sizeof(chunk_t));
    if (!h) {
        return NULL;
    }

    h->varcalls = init_ta_v();
    if (!h->varcalls) {
        free(h);
        return NULL;
    }

    kv_init(h->reads);

    h->qnames = (kstring_t **)calloc(init_size, sizeof(kstring_t *));
    if (!h->qnames) {
        kv_destroy(h->reads);
        destroy_ta_v(h->varcalls);
        free(h);
        return NULL;
    }

    h->qname2ID = kh_init(htstri_t);

    h->abs_start = start;
    h->abs_end = end;
    h->compat0 = 0;
    return h;
}

static void push_read_chunk_t(chunk_t *h, read_t read, const char *qname) {
    // get a slot for incoming read
    size_t m = h->reads.m;
    kv_push(read_t, h->reads, read);
    if (h->reads.m != m) {  // realloc'd
        h->qnames = (kstring_t **)realloc(h->qnames, sizeof(kstring_t *) * h->reads.m);
    }
    assert(h->reads.n > 0);
    h->qnames[h->reads.n - 1] = (kstring_t *)calloc(1, sizeof(kstring_t));
    assert(h->qnames[h->reads.n - 1]);
    ksprintf(h->qnames[h->reads.n - 1], "%s", qname);
}

static inline unsigned char filter_base_by_qv(char raw, int min_qv) {
    return (int)raw - 33 >= min_qv ? raw : 'N';
}

static int parse_variants_for_one_read(const bam1_t *aln,
                                       qa_v *vars,
                                       int min_base_qv,
                                       uint32_t *left_clip_len,
                                       uint32_t *right_clip_len) {
    // note: caller ensure that MD tag exists and
    //       does not have unexpected operations.
    // Return: 0 if ok, 1 when error
    const int SNPonly = 0;
    int debug_print = 0;
    int failed = 0;

    int self_start = 0;
    const uint32_t ref_start = static_cast<uint32_t>(aln->core.pos);

    // parse cigar for insertions
    vu64_t insertions;  // for parsing MD tag
    kv_init(insertions);
    uint32_t *cigar = bam_get_cigar(aln);
    uint32_t op, op_l;
    char *seq = (char *)malloc(16);
    assert(seq);
    uint32_t seq_n = 16;
    uint8_t *seqi = bam_get_seq(aln);
    uint32_t ref_pos = ref_start;
    uint32_t self_pos = 0;
    if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
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
            if (op_l >= seq_n) {
                seq_n = op_l + 1;
                seq = (char *)realloc(seq, seq_n);
            }
            for (uint32_t j = 0; j < op_l; j++) {
                seq[j] = filter_base_by_qv(seq_nt16_str[bam_seqi(seqi, self_pos + j)], min_base_qv);
            }
            if (!SNPonly) {
                add_allele_qa_v(vars, ref_pos, seq, op_l, VAR_OP_I);  // push_to_vvar_t()
            }
            kv_push(uint64_t, insertions, ((uint64_t)op_l) << 32 | self_pos);
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
    uint8_t *tagd = bam_aux_get(aln, "MD");
    char *md_s = bam_aux2Z(tagd);
    int prev_md_i = 0;
    int prev_md_type, md_type;
    if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
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
                while (prev_ins_idx < insertions.n &&
                       self_pos > (uint32_t)insertions.a[prev_ins_idx]) {
                    self_pos += insertions.a[prev_ins_idx] >> 32;
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
                if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
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

    if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
        for (size_t tmpi = 0; tmpi < vars->n; tmpi++) {
            int tmpop = vars->a[tmpi].allele.a[vars->a[tmpi].allele.n - 2];
            fprintf(stderr, "[dbg::%s]    op=%c pos=%d len=%d ", __func__, "MXID"[tmpop],
                    vars -> a[tmpi].pos, (int)vars->a[tmpi].allele.n - 1);
            if (tmpop > 0) {
                fprintf(stderr, "seq=");
                for (size_t tmpj = 0; tmpj < vars->a[tmpi].allele.n - 1; tmpj++) {
                    fprintf(stderr, "%c", "ACGT?"[vars->a[tmpi].allele.a[tmpj]]);
                }
            }
            fprintf(stderr, "\n");
        }
    }

    kv_destroy(insertions);
    free(seq);

    return failed;
}

typedef struct {
    // LIMITATION: number of reads is at most (1<<27)-1
    uint64_t key;    // pos:32 | cigar:4 | strand:1 | readID:27
    uint32_t varID;  // as in the read
} pg_t;              // piggyback
typedef kvec_t(pg_t) pg_v;
#define pg_t_cmp(x, y) ((x).key < (y).key)
void ks_mergesort_kspg(size_t n, pg_t array[], pg_t temp[]);
pg_t ks_ksmall_kspg(size_t n, pg_t arr[], size_t kk);
void ks_shuffle_kspg(size_t n, pg_t a[]);
void ks_sample_kspg(size_t n, size_t r, pg_t a[]);
KSORT_INIT(kspg, pg_t, pg_t_cmp)
static void push_to_pg_t(pg_v *pg, uint32_t i_read, uint32_t pos, uint8_t op, uint32_t i_var) {
    if (i_read >= (1UL << 27)) {
        fprintf(stderr, "[E::%s] i_read value does not fit into 27 bits. iread = %u\n", __func__,
                i_read);
        return;
    }
    uint64_t key = ((uint64_t)pos) << 32 | ((uint64_t)op) << 28 | ((uint64_t)op) << 27 | i_read;
    pg_t tmp;
    tmp.key = key;
    tmp.varID = i_var;
    kv_push(pg_t, *pg, tmp);
}

static int sort_qa_v_for_all(chunk_t *ck) {
    int tot = 0;
    qa_v *buf = init_qa_v();
    dummyexpand_qa_v(buf, 16);
    for (size_t i = 0; i < ck->reads.n; i++) {
        tot += sort_qa_v(ck->reads.a[i].vars, buf);
    }
    kv_destroy(*buf);  // dummy buffer does not have the allele slots,
                       // don't call the destroy method
    free(buf);
    return tot;
}

static int get_cigar_op_qa_t(const qa_t *h) { return h->allele.a[h->allele.n - 1]; }

static void log_allele_indices_for_reads_given_varcalls(chunk_t *ck) {
    // After pileup/unphased varcall, collect indices of
    // alleles of interest on each read (stored sorted wrt to variant's
    // reference positions), so that we will not need
    // to refer to actual allele sequences when constructing
    // the variant graph.

    sort_qa_v_for_all(ck);

    ta_v *vars = ck->varcalls;
    for (size_t i_read = 0; i_read < ck->reads.n; i_read++) {
        read_t *read = &ck->reads.a[i_read];
        qa_v *new_ = init_qa_v();
        size_t prev_i = 0;
        for (uint32_t i_pos = 0; i_pos < vars->n; i_pos++) {
            uint32_t pos = vars->a[i_pos].pos;
            ta_t *var = &vars->a[i_pos];
            if (!var->is_used) {
                continue;
            }
            if (pos < read->start_pos || pos >= read->end_pos) {
                continue;
            }
            for (size_t i = prev_i; i < read->vars->n; i++) {
                qa_t *readvar = &read->vars->a[i];
                if (readvar->pos > pos) {
                    break;
                }
                if (readvar->pos == pos) {
                    int allele_idx = get_allele_index(var, &readvar->allele);
                    if (allele_idx < 0 || !var->alleles_is_used.a[allele_idx]) {
                        continue;
                    }

                    // if we reach here, current position and self allele is in use
                    add_allele_qa_v_nt4seq(
                            new_, pos, &readvar->allele,
                            static_cast<int>(readvar->allele.n) - 1,  // last slot is cigar op
                            readvar->allele.a[readvar->allele.n - 1]);

                    readvar = &new_->a[new_->n - 1];

                    // update flags and reverse index entries
                    readvar->is_used = 1;
                    readvar->hp = HAPTAG_UNPHASED;
                    readvar->var_idx = i_pos;
                    readvar->allele_idx = allele_idx;

                    // step forward
                    prev_i = i + 1;
                    break;
                }
            }
        }
        // dealloc and replace with sorted, indexed entries
        destroy_qa_v(read->vars);
        read->vars = new_;
    }
}

static void collect_raw_read_compatibility(chunk_t *ck) {
    // Should be called prior to informative site identification
    // (but with read variants all sorted by coordinates)
    // to rule out weird large SVs, since uninformative variants
    // are hard-removed after that .
    // TODO: is n^2, ok for wgs local but rnaseq/amplicon maybe not so much.
    int debug_print = 0;

    if (!ck->compat0) {
        ck->compat0 = (uint8_t **)calloc(ck->reads.n * ck->reads.n, sizeof(uint8_t *));
        assert(ck->compat0);
        for (size_t i = 0; i < ck->reads.n; i++) {
            ck->compat0[i] = (uint8_t *)calloc(ck->reads.n, 1);
            assert(ck->compat0[i]);
        }
    }
    for (size_t i = 0; i < ck->reads.n; i++) {
        for (size_t j = 0; j < ck->reads.n; j++) {
            ck->compat0[i][j] = CMPT0_COMPAT;
        }
    }

    // mark non overlapping reads as incompatible
    for (size_t i = 0; i < ck->reads.n - 1; i++) {
        for (size_t j = i + 1; j < ck->reads.n; j++) {
            if (ck->reads.a[j].start_pos >= ck->reads.a[i].end_pos) {
                ck->compat0[i][j] = CMPT0_INCOMPAT;
                ck->compat0[j][i] = CMPT0_INCOMPAT;
            }
        }
    }
    if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
        FILE *fp = fopen("test.compat_raw", "w");
        for (size_t i = 0; i < ck->reads.n - 1; i++) {
            for (size_t j = i; j < ck->reads.n; j++) {
                fprintf(fp, "%s\t%s\t%d\n", ck->qnames[i]->s, ck->qnames[j]->s, ck->compat0[i][j]);
            }
        }
        fclose(fp);
    }
}

static void update_read_compatibility(chunk_t *ck) {
    // should be called by the end of unphased varcall.
    int debug_print = 0;
    int n_updated = 0;
    uint32_t n_reads = static_cast<uint32_t>(ck->reads.n);
    for (uint32_t i = 0; i < n_reads - 1; i++) {
        if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr, "[dbg::%s] at read #%d/%d\n", __func__, i, n_reads);
        }
        for (uint32_t j = i + 1; j < n_reads; j++) {
            if (ck->reads.a[j].start_pos < ck->reads.a[i].end_pos) {
                if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
                    fprintf(stderr, "[dbg::%s] check compat: %s(i=%d) and %s(j=%d)\n", __func__,
                            ck->qnames[i]->s, i, ck->qnames[j]->s, j);
                }
                if (ck->compat0[i][j] & CMPT0_COMPAT) {
                    int is_compat =
                            check_if_two_reads_are_compatible(&ck->reads.a[i], &ck->reads.a[j], 0);
                    if (is_compat <= 0) {
                        ck->compat0[i][j] = CMPT0_INCOMPAT;
                        ck->compat0[j][i] = CMPT0_INCOMPAT;
                        n_updated++;
                    }
                }
            } else {
                ck->compat0[i][j] = CMPT0_AWAY;
                ck->compat0[j][i] = CMPT0_AWAY;
            }
        }
    }
    if (DEBUG_LOCAL_HAPLOTAGGING) {
        fprintf(stderr, "[M::%s] updated %d pairs of %d reads\n", __func__, n_updated,
                (int)ck->reads.n);
    }
}

static void mark_contained_reads(chunk_t *ck, uint32_t overhang_threshold) {
    // Abuse ck->compat0 and mark on the diagnal when a read
    // is contained by some other read.
    int debug_print = 0;
    int n = 0;
    for (size_t i = 0; i < ck->reads.n - 1; i++) {
        if (ck->compat0[i][i] & CMPT0_F) {
            continue;
        }
        uint32_t start0 = ck->reads.a[i].start_pos;
        uint32_t end0 = ck->reads.a[i].end_pos;
        for (size_t j = 0; j < ck->reads.n; j++) {
            if (j == i) {
                continue;
            }
            if (ck->reads.a[i].left_clip_len >= overhang_threshold ||
                ck->reads.a[i].right_clip_len >= overhang_threshold ||
                ck->reads.a[j].left_clip_len >= overhang_threshold ||
                ck->reads.a[j].right_clip_len >= overhang_threshold) {
                continue;
            }
            if (ck->compat0[j][j] & CMPT0_F) {
                continue;
            }
            if (ck->compat0[j][i] & CMPT0_F) {
                continue;
            }
            if (ck->compat0[i][j] & CMPT0_F) {
                continue;
            }
            uint32_t start1 = ck->reads.a[j].start_pos;
            uint32_t end1 = ck->reads.a[j].end_pos;
            if (start0 <= start1 && end0 > end1) {
                ck->compat0[j][j] = CMPT0_CONTAINED;
                if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
                    fprintf(stderr, "[dbg::%s] shadow %s (due to %s)\n", __func__, ck->qnames[j]->s,
                            ck->qnames[i]->s);
                }
            } else if (start1 >= start0 && end1 < end0) {
                ck->compat0[i][i] = CMPT0_CONTAINED;
                if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
                    fprintf(stderr, "[dbg::%s] shadow %s (due to %s)\n", __func__, ck->qnames[i]->s,
                            ck->qnames[j]->s);
                }
            }
        }
    }
    for (size_t i = 0; i < ck->reads.n; i++) {
        if (ck->compat0[i][i] & CMPT0_CONTAINED) {
            n++;
        }
    }
    if (DEBUG_LOCAL_HAPLOTAGGING) {
        fprintf(stderr, "[M::%s] marked %d reads as contained\n", __func__, n);
    }
}

static vu64_t *TRF_heuristic(const char *seq, int seq_l, int ref_start) {
    // A simple tandem repeat masker inspired by TRF.
    // Uses arbitrary thresholds and does not exclude
    // non-tandem direct repeats, unlike the TRF.
    // The usecase is to merely avoid picking up variants
    // too close to low-complexity runs for phasing.
    int debug_print = 0;
    if (DEBUG_LOCAL_HAPLOTAGGING) {
        fprintf(stderr, "[dbg::%s] seq_l is %d, offset %d\n", __func__, seq_l, ref_start);
    }
    FILE *fp = 0;
    if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
        fp = fopen("test.bed", "w");
        assert(fp);
    }

    // length for exact match lenth
    int k = 5;  // note:  we will enumerate all combos

    // input sancheck
    if (seq_l < k * 3) {  // sequence too short, nothing to do
        return NULL;
    }

    // init buffer for kmer index
    uint32_t idx_l = 1;
    for (int i = 0; i < k; i++) {
        idx_l *= 4;
    }
    vu32_t *idx = (vu32_t *)malloc(sizeof(vu32_t) * idx_l);
    assert(idx);
    for (uint32_t i = 0; i < idx_l; i++) {
        kv_init(idx[i]);
    }
    uint8_t *idx_good = (uint8_t *)calloc(idx_l, 1);
    assert(idx_good);
    if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
        fprintf(stderr, "[dbg::%s] idx_l %d\n", __func__, (int)idx_l);
    }

    // index kmers
    uint32_t mer = 0;
    uint32_t mermask = (1 << (k * 2)) - 1;
    int mer_n = 0;
    for (int i = 0; i < seq_l; i++) {
        int c = (int)kdy_seq_nt4_table[(uint8_t)seq[i]];
        if (c != 4) {
            mer = (mer << 2 | c) & mermask;
            if (mer_n < k) {
                mer_n++;
            }
            if (mer_n == k) {
                vu32_t *p = &idx[mer];
                kv_push(uint32_t, *p, i + ref_start - k + 1);  // use absolute coordinate
            }
        } else {
            mer = 0;
            mer_n = 0;
        }
    }
    if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
        for (uint32_t i = 0; i < idx_l; i++) {
            if (idx[i].n != 0) {
                fprintf(stderr, "[dbg::%s] combo#%d ", __func__, i);
                for (int j = 0; j < k; j++) {
                    fprintf(stderr, "%c", "ACGT"[(i >> ((k - j - 1) * 2)) & 3]);
                }
                fprintf(stderr, " n=%d\n", (int)idx[i].n);
                if (debug_print > 1) {
                    fprintf(stderr, "   ^");
                    for (size_t j = 0; j < idx[i].n; j++) {
                        fprintf(stderr, " %d", idx[i].a[j]);
                    }
                    fprintf(stderr, "\n");
                }
            }
        }
    }

    // calculate distances between matches
    uint32_t **dists = (uint32_t **)malloc(sizeof(uint32_t *) *
                                           idx_l);  // collects kmer match intervals of each kmer
    assert(dists);
    vu32_t ds0;  // collects non-singleton kmer match intervals from different kmers
    kv_init(ds0);
    for (uint32_t i = 0; i < idx_l; i++) {
        if (idx[i].n >= 3) {
            const int l = static_cast<int>(idx[i].n);
            idx_good[i] = 1;
            dists[i] = (uint32_t *)malloc(sizeof(uint32_t) * l);
            assert(dists[i]);
            dists[i][0] = 0;
            for (int j = 1; j < l; j++) {
                dists[i][j] = idx[i].a[j] - idx[i].a[j - 1];
            }

            // sort interval sizes
            radix_sort_ksu32(dists[i], dists[i] + l);

            // Is there any recurring interval sizes?
            // if yes, remember them; otherwise blacklist the kmer.
            int ok = 0;
            int new_ = 1;
            for (int j = 2; j < l; j++) {
                if (dists[i][j] == dists[i][j - 1] && dists[i][j] == dists[i][j - 2]) {
                    if (new_) {
                        ok = 1;
                        new_ = 0;
                        kv_push(uint32_t, ds0, dists[i][j]);
                    }
                }
            }
            if (!ok) {
                idx_good[i] = 0;
                free(dists[i]);
            } else {  // looks ok, restore the order in the buffer, we will use it again
                for (int j = 1; j < l; j++) {
                    dists[i][j] = idx[i].a[j] - idx[i].a[j - 1];
                }
            }
        }
    }

    // find promising match interval sizes
    vu32_t ds;
    kv_init(ds);
    radix_sort_ksu32(ds0.a, ds0.a + ds0.n);
    int cnt = 0;
    for (size_t i = 1; i < ds0.n; i++) {
        if (ds0.a[i] != ds0.a[i - 1]) {
            if (cnt > 0) {
                if (ds0.a[i - 1] < TRF_MOTIF_MAX_LEN) {
                    kv_push(uint32_t, ds, ds0.a[i - 1]);
                }
            }
            cnt = 0;
        } else {
            cnt++;
        }
    }
    if (cnt > 0 && ds0.n > 0 && ds0.a[ds0.n - 1] < TRF_MOTIF_MAX_LEN) {
        kv_push(uint32_t, ds, ds0.a[ds0.n - 1]);
    }
    vu64_t *intervals = 0;
    if (ds.n == 0) {
        if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr, "[dbg::%s] ds is empty (ds0 was %d)\n", __func__, (int)ds0.n);
            for (size_t i = 0; i < ds0.n; i++) {
                fprintf(stderr, "[dbg::%s] ds0 #%d is %d\n", __func__, (int)i, (int)ds0.a[i]);
            }
        }

    } else {
        if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr, "[dbg::%s] there are %d candidate distances (raw: %d)\n", __func__,
                    (int)ds.n, (int)ds0.n);
            for (size_t i = 0; i < ds.n; i++) {
                fprintf(stderr, "[dbg::%s] ds #%d is %d\n", __func__, (int)i, (int)ds.a[i]);
            }
        }

        // process the kmer chains
        intervals = (vu64_t *)calloc(1, sizeof(vu64_t));
        assert(intervals);
        kv_init(*intervals);
        vu32_t poss;
        vu32_t intervals_buf;
        kv_init(poss);
        kv_init(intervals_buf);
        int buf[3] = {0, 0, 0};
        uint8_t *used = (uint8_t *)calloc(idx_l, 1);
        assert(used);
        int failed = 0;
        for (size_t i_d = 0; i_d < ds.n; i_d++) {
            uint32_t d = ds.a[i_d];
            for (uint32_t i_mer = 0; i_mer < idx_l; i_mer++) {
                if (!idx_good[i_mer]) {
                    continue;
                }

                uint8_t mer_is_used = 0;

                poss.n = 0;
                int ok = 0;
                int stop = 0;
                for (size_t i = 0; i < idx[i_mer].n;
                     i++) {  // require at least 1 hit for three consecutive kmers
                    if (i + 3 < idx[i_mer].n) {
                        buf[0] = dists[i_mer][i + 1] == d;
                        buf[1] = dists[i_mer][i + 2] == d;
                        buf[2] = dists[i_mer][i + 3] == d;
                        ok = buf[0] || buf[1] || buf[2];
                        buf[0] = dists[i_mer][i + 1] >= d && dists[i_mer][i + 1] < 3 * d + 3;
                        buf[1] = dists[i_mer][i + 2] >= d && dists[i_mer][i + 2] < 3 * d + 3;
                        buf[2] = dists[i_mer][i + 3] >= d && dists[i_mer][i + 3] < 3 * d + 3;
                        ok = ok & buf[0] & buf[1] & buf[2];
                        if (poss.n == 0 && !buf[0]) {
                            ok = 0;
                        }
                        if (ok) {
                            kv_push(uint32_t, poss, (uint32_t)i);
                        }
                    } else if (i + 2 == idx[i_mer].n - 1) {
                        if (dists[i_mer][i + 1] == d || dists[i_mer][i + 2] == d) {
                            kv_push(uint32_t, poss, (uint32_t)i);
                            ok = 1;
                        } else {
                            ok = 0;
                        }
                    } else if (i + 1 == idx[i_mer].n - 1) {
                        if (dists[i_mer][i + 1] == d) {
                            kv_push(uint32_t, poss, (uint32_t)i);
                            kv_push(uint32_t, poss, (uint32_t)i + 1);
                            ok = 1;
                            stop = 1;
                        } else if (dists[i_mer][i] == d) {
                            kv_push(uint32_t, poss, (uint32_t)i);
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
                        if (poss.n > 0) {
                            while (poss.n > 0 && dists[i_mer][poss.a[poss.n - 1]] != d) {
                                poss.n--;
                            }
                            int start = idx[i_mer].a[poss.a[0]];
                            int end = idx[i_mer].a[poss.a[poss.n == 0 ? 0 : poss.n - 1]];
                            if (end - start > k + 2) {
                                if (start > TRF_ADD_PADDING + 1) {
                                    start -= k + TRF_ADD_PADDING + 1;
                                }
                                end += 2 * k + TRF_ADD_PADDING;
                                kv_push(uint32_t, intervals_buf, start << 1);
                                kv_push(uint32_t, intervals_buf, end << 1 | 1);
                                if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
                                    fprintf(stderr, "[dbg::%s] d=%d mer=%d saw start-end: %d-%d\n",
                                            __func__, d, i_mer, start, end);
                                }
                                mer_is_used = 1;
                            }
                        }
                        poss.n = 0;
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
        kv_destroy(poss);
        free(used);

        // merge intervals (requires a minimum depth)
        if (failed /*impossible seen when parsing mers*/
            || intervals_buf.n == 0) {
            if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
                fprintf(stderr,
                        "[dbg::%s] failed parsing mers (%d) or intervals_buf is empty (%d)\n",
                        __func__, failed, (int)intervals_buf.n);
            }
            kv_destroy(*intervals);
            intervals = NULL;
            kv_destroy(intervals_buf);

        } else {
            radix_sort_ksu32(intervals_buf.a, intervals_buf.a + intervals_buf.n);
            int depth = 0;
            int start = -1;
            int end = -1;
            int prev_pos = -1;
            for (size_t i = 0; i < intervals_buf.n; i++) {
                if (intervals_buf.a[i] & 1) {
                    if (depth > 0) {
                        depth--;
                    }
                } else {
                    depth++;
                }

                int current_pos = intervals_buf.a[i] >> 1;
                const int lbreak = 0;

                if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
                    fprintf(stderr, "[dbg::%s] %d stat=%d depth=%d lbreak=%d\n", __func__,
                            current_pos, intervals_buf.a[i] & 1, depth, lbreak);
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
                            if (intervals->n > 0) {
                                if (start - (uint32_t)intervals->a[intervals->n - 1] <
                                    TRF_CLOSE_GAP_THRESHOLD) {  // merge block
                                    intervals->a[intervals->n - 1] =
                                            (intervals->a[intervals->n - 1] >> 32) << 32;
                                    intervals->a[intervals->n - 1] |= end;
                                } else {  // new block
                                    kv_push(uint64_t, *intervals, ((uint64_t)start) << 32 | end);
                                }
                            } else {  // new block
                                kv_push(uint64_t, *intervals, ((uint64_t)start) << 32 | end);
                            }
                        }
                        start = lbreak ? current_pos : -1;
                        end = -1;
                    }
                }
                if (lbreak) {
                    depth = 0;
                }
                prev_pos = intervals_buf.a[i] >> 1;
            }
            if (start != -1) {
                end = intervals_buf.a[intervals_buf.n - 1] >> 1;
                if (end > start) {
                    // merge small gaps
                    if (intervals->n > 0) {
                        if (start - (uint32_t)intervals->a[intervals->n - 1] <
                            TRF_CLOSE_GAP_THRESHOLD) {  // merge block
                            intervals->a[intervals->n - 1] = (intervals->a[intervals->n - 1] >> 32)
                                                             << 32;
                            intervals->a[intervals->n - 1] |= end;
                        } else {  // new block
                            kv_push(uint64_t, *intervals, ((uint64_t)start) << 32 | end);
                        }
                    } else {  // new block
                        kv_push(uint64_t, *intervals, ((uint64_t)start) << 32 | end);
                    }
                }
            }
            kv_destroy(intervals_buf);
            if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
                fprintf(stderr, "[dbg::%s] had %d intervals\n", __func__, (int)intervals->n);
                for (size_t i = 0; i < intervals->n; i++) {
                    fprintf(stderr, "[dbg::%s]    #%d %d-%d\n", __func__, (int)i,
                            (int)(intervals->a[i] >> 32), (int)(uint32_t)intervals->a[i]);
                }
            }

            if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
                for (size_t i = 0; i < intervals->n; i++) {
                    fprintf(fp, "chr20\t%d\t%d\n", (int)(intervals->a[i] >> 32),
                            (int)(uint32_t)intervals->a[i]);
                }
            }
        }
    }

    for (uint32_t i = 0; i < idx_l; i++) {
        kv_destroy(idx[i]);
        if (idx_good[i]) {
            free(dists[i]);
        }
    }
    free(idx_good);
    free(idx);
    free(dists);
    kv_destroy(ds0);
    kv_destroy(ds);
    if (debug_print) {
        fclose(fp);
    }
    return intervals;
}

static vu64_t *get_lowcmp_mask(faidx_t *fai, const char *refname, int ref_start, int ref_end) {
    char *refseq_s = 0;
    int refseq_l = 0;
    const int span_sl = get_posint_digits(ref_start) + get_posint_digits(ref_end) +
                        static_cast<int>(strlen(refname)) + 3;
    char *span_s = (char *)malloc(span_sl);
    if (!span_s) {
        if (DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr, "[E::%s] alloc failed (span_s)\n", __func__);
        }
        return NULL;
    }

    snprintf(span_s, span_sl, "%s:%d-%d", refname, ref_start + 1, ref_end);
    refseq_s = fai_fetch(fai, span_s, &refseq_l);
    free(span_s);

    if (refseq_l <= 0 || !refseq_s) {
        if (DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr,
                    "[W::%s] failed to fetch reference sequence (refname=%s, start=%d, end=%d; err "
                    "code %d)\n",
                    __func__, refname, ref_start, ref_end, refseq_l);
        }
        if (refseq_s) {
            hts_free(refseq_s);
        }
        return NULL;
    }
    vu64_t *ret = TRF_heuristic(refseq_s, refseq_l, ref_start);
    if (refseq_s) {
        hts_free(refseq_s);
    }
    return ret;
}

chunk_t *unphased_varcall_a_chunk(bamfile_t *hf,
                                  ref_vars_t *refvars,
                                  faidx_t *ref_faidx,
                                  const char *refname,
                                  int32_t itvl_start,
                                  int32_t itvl_end,
                                  int itvl_disable_expansion,
                                  int min_base_quality,
                                  int min_varcall_coverage,
                                  float min_varcall_coverage_ratio,
                                  int varcall_indel_mask_flank,
                                  int disable_info_site_filtering,
                                  int need_to_collect_compat,
                                  int max_clipping,
                                  int max_reads_in_chunk) {
    int SNPonly = 1;
    int debug_print = 0;
    int failed = 0;

    const int noflt = disable_info_site_filtering;
    const int is_using_lasm = need_to_collect_compat;
    const int min_mapq = 10;
    const float max_aln_de = static_cast<float>(READ_MAX_DIVERG);
    const float min_het_ratio = 0.1f;  // e.g. 0.1 means only take positions with
                                       // non-ref allele frequencies summed to [0.1, 0.9)
                                       // as possibly heterozygous.

    int base_q_min = noflt ? 0 : min_base_quality;
    uint32_t var_call_min_cov = noflt ? 0 : min_varcall_coverage < 0 ? 0 : min_varcall_coverage;
    float var_call_min_cov_ratio = noflt ? 0 : min_varcall_coverage_ratio;
    int var_call_min_strand_presence = noflt ? 0 : VAR_CALL_MIN_STRAND_PRESENCE;
    uint32_t var_call_neigh_indel_flank = noflt                          ? 0
                                          : varcall_indel_mask_flank < 0 ? 0
                                                                         : varcall_indel_mask_flank;

    chunk_t *ck = init_chunk_t(itvl_start, itvl_end);
    if (!ck) {
        return NULL;
    }

    const size_t itvl_len = strlen(refname) + get_posint_digits(itvl_end) * 2 + 5;
    char *itvl = (char *)calloc(itvl_len, 1);
    assert(itvl);
    snprintf(itvl, itvl_len, "%s:%d-%d", refname, itvl_start, itvl_end);
    hts_itr_t *bamitr = sam_itr_querys(hf->bai, hf->header, itvl);
    bam1_t *aln = bam_init1();  // use local buffer instead of hf->aln
                                // TODO: can use hf->aln, the bam file isn't
                                // thread safe and impl here has ended up open bam in
                                // each threads...

    int absent;
    uint32_t readID = 0;
    qa_v *tmp_qav = init_qa_v();
    qa_v *tmp_qavsort = init_qa_v();
    dummyexpand_qa_v(tmp_qavsort, 16);

    // Adjust region start and end: if there happens to be no
    // heterozygous variants around start and/or end, we could have
    // 30-50% unread unphased. Here, we allow expanding the
    // target region *somewhat*: the start/end of the first/last
    // read, if they are not too long; or +-50kb when they are
    // too long.
    uint32_t abs_start = ck->abs_start;
    uint32_t abs_end = ck->abs_end;
    int offset_default = 50000;
    if (!itvl_disable_expansion) {
        int offset_left = offset_default;
        int offset_right = 0;

        // left
        snprintf(itvl, itvl_len, "%s:%d-%d", refname, itvl_start, itvl_start + 1);
        hts_itr_destroy(bamitr);
        bamitr = sam_itr_querys(hf->bai, hf->header, itvl);
        while (sam_itr_next(hf->fp, bamitr, aln) >= 0) {
            if (itvl_start - aln->core.pos < offset_default) {
                offset_left = static_cast<int>(itvl_start - aln->core.pos);
                break;
            }
        }

        //right
        snprintf(itvl, itvl_len, "%s:%d-%d", refname, itvl_end, itvl_end + 1);
        hts_itr_destroy(bamitr);
        bamitr = sam_itr_querys(hf->bai, hf->header, itvl);
        while (sam_itr_next(hf->fp, bamitr, aln) >= 0) {
            int end_pos = (int)bam_endpos(aln);
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
        snprintf(itvl, itvl_len, "%s:%d-%d", refname, abs_start, abs_end);
        hts_itr_destroy(bamitr);
        bamitr = sam_itr_querys(hf->bai, hf->header, itvl);
        if (DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr, "[M::%s] interval expansion: now using %s (-%d, +%d)\n", __func__, itvl,
                    offset_left, offset_right);
        }
    }

    uint64_t n_reads = 0;
    while (sam_itr_next(hf->fp, bamitr, aln) >= 0) {
        n_reads++;
        if (n_reads > MAX_READS) {
            if (DEBUG_LOCAL_HAPLOTAGGING) {
                fprintf(stderr, "[E::%s] too many reads\n", __func__);
            }
            failed = 1;
            break;
        }
        char *qn = bam_get_qname(aln);
        // some mapping quality filtering
        int flag = aln->core.flag;
        int mapq = (int)aln->core.qual;
        float de = 0;
        uint8_t *tmp = bam_aux_get(aln, "de");
        if (tmp) {
            de = static_cast<float>(bam_aux2f(tmp));
        }

        // require MD tag to exist and does not have unexpected operations
        int md_is_ok = sancheck_MD_tag_exists_and_is_valid(aln);
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

        // the buffer
        read_t read;
        init_read_t(&read);
        push_read_chunk_t(ck, read, qn);

        // read info
        khint_t key = kh_put(htstri_t, ck->qname2ID, ck->qnames[readID]->s, &absent);
        if (!absent) {
            if (DEBUG_LOCAL_HAPLOTAGGING) {
                fprintf(stderr, "[W::%s] dup read name? qn=%s\n", __func__, qn);
            }
        } else {
            kh_key(ck->qname2ID, key) = ck->qnames[readID]->s;
            kh_val(ck->qname2ID, key) = readID;
        }

        // collect variants of the read
        read_t *r = &ck->reads.a[readID];
        r->ID = readID;
        r->start_pos = static_cast<uint32_t>(aln->core.pos);
        r->end_pos = static_cast<uint32_t>(bam_endpos(aln));
        r->strand = !!(flag & 16);
        r->de = de;
        if (!refvars) {
            int parse_failed = parse_variants_for_one_read(aln, r->vars, base_q_min,
                                                           &r->left_clip_len, &r->right_clip_len);
            if (parse_failed) {
                destroy_read_t(r, 0);
                // and do not increment readID; we want to pretend this read doesn't exit
            } else {
                sort_qa_v(r->vars, tmp_qavsort);
                readID++;
            }
        } else {
            wipe_qa_v(tmp_qav);  // TODO: maybe add a flag to avoid repeated de/re-alloc?
            int parse_failed = parse_variants_for_one_read(aln, tmp_qav, base_q_min,
                                                           &r->left_clip_len, &r->right_clip_len);
            if (parse_failed) {
                destroy_read_t(r, 0);
                // and do not increment readID; we want to pretend this read doesn't exit
            } else {
                sort_qa_v(tmp_qav, tmp_qavsort);
                filter_lift_qa_v_given_conf_list(tmp_qav, r->vars, refvars);
                readID++;
            }
        }
        if (ck->reads.n > (size_t)max_reads_in_chunk) {  // the user-imposed limit
            failed = 1;
            break;
        }
    }
    free(itvl);
    destroy_qa_v(tmp_qav);
    kv_destroy(*tmp_qavsort);
    free(tmp_qavsort);
    bam_destroy1(aln);
    hts_itr_destroy(bamitr);

    if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
        fprintf(stderr, "[dbg::%s] chunk has %d reads\n", __func__, (int)ck->reads.n);
    }

    if (ck->reads.n < 2 || failed) {
        destroy_chunk_t(ck, 1);
        return NULL;
    }

    if (need_to_collect_compat) {  // lasm (wip)
        collect_raw_read_compatibility(ck);
    }

    // collect hp/lowcmp mask
    vu64_t *hplowcmp_mask = get_lowcmp_mask(ref_faidx, refname, abs_start, abs_end);

    // piggyback sorting pileup 1st pass: without ref calls
    uint64_t mask_readID = ~(UINT64_MAX << 27);
    pg_v pg;
    kv_init(pg);
    pg_v pg_buf;
    kv_init(pg_buf);
    for (uint32_t i_read = 0; i_read < ck->reads.n; i_read++) {
        qa_v *vars = ck->reads.a[i_read].vars;
        for (uint32_t i_pos = 0; i_pos < vars->n; i_pos++) {
            qa_t *var = &vars->a[i_pos];

            if (var->pos < abs_start || var->pos >= abs_end) {
                continue;
            }
            if (SNPonly && get_cigar_op_qa_t(var) != VAR_OP_X) {
                continue;
            }
            int is_banned = 0;
            if (hplowcmp_mask) {
                for (size_t i = 0; i < hplowcmp_mask->n; i++) {
                    if (var->pos >= hplowcmp_mask->a[i] >> 32 &&
                        var->pos < (uint32_t)hplowcmp_mask->a[i]) {
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
            for (size_t i = 0; i < var->allele.n - 1; i++) {
                if (var->allele.a[i] > 3) {
                    is_sus = 1;
                    break;
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
                    qa_t *var_l = &vars->a[i_pos - 1];
                    const uint8_t var_l_op = static_cast<uint8_t>(get_cigar_op_qa_t(var_l));
                    if (var->pos - var_l->pos <= var_call_neigh_indel_flank &&
                        var_l_op == VAR_OP_I) {
                        continue;
                    }
                    if (var_l_op == VAR_OP_D) {  // (for del, use end position of the del run
                                                 // rather than start position.)
                        const uint32_t cigar_l = static_cast<uint32_t>(var_l->allele.n - 1);
                        if (var->pos - (var_l->pos + cigar_l) <= var_call_neigh_indel_flank) {
                            continue;
                        }
                    }
                }
                if (i_pos < vars->n - 1) {
                    qa_t *var_r = &vars->a[i_pos + 1];
                    if (var_r->pos - var->pos <= var_call_neigh_indel_flank &&
                        get_cigar_op_qa_t(var_r) & (VAR_OP_D | VAR_OP_I)) {
                        continue;
                    }
                }
            }

            push_to_pg_t(&pg, i_read, var->pos, static_cast<uint8_t>(get_cigar_op_qa_t(var)),
                         i_pos);
        }
    }
    if (hplowcmp_mask) {
        kv_destroy(*hplowcmp_mask);
        free(hplowcmp_mask);
    }

    if (pg.n == 0) {
        kv_destroy(pg);
        kv_destroy(pg_buf);
        if (DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr, "[M::%s] no variant to call (%s:%d-%d)\n", __func__, refname,
                    ck->abs_start, ck->abs_end);
        }
        destroy_chunk_t(ck, 1);
        return NULL;
    }
    kv_resize(pg_t, pg_buf, pg.n);
    ks_mergesort_kspg(pg.n, pg.a, pg_buf.a);

    int has_ref_allele = 0;

    // prepare the 2nd piggyback sorting: add in ref calls
    // for positions that may be het.
    // If we collect ref later it would a pita having to have
    // an index sentinel. Better to have an allele sequence sentinel instead.
    has_ref_allele = 1;
    int n_added_ref = 0;
    size_t pgn = pg.n;
    int strand_count[2];
    int strand;
    for (size_t i = 0; i < pgn - 1; /******/) {
        uint32_t pos = pg.a[i].key >> 32;
        strand_count[0] = 0;
        strand_count[1] = 0;
        strand = !!(ck->reads.a[pg.a[i].key & mask_readID].strand);
        strand_count[strand]++;

        // check if has enough alts and fair read strands
        size_t j;
        for (j = i + 1; j < pg.n; j++) {
            if ((pg.a[j].key >> 32) != pos) {
                break;
            }
            strand = !!(ck->reads.a[pg.a[j].key & mask_readID].strand);
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
        for (size_t i_read = 0; i_read < ck->reads.n; i_read++) {  // TODO: is this slow
            if (ck->reads.a[i_read].start_pos <= pos && ck->reads.a[i_read].end_pos > pos) {
                cov_tot++;
            }
        }
        float call_ratio = (float)(j - i) / cov_tot;
        if (call_ratio < min_het_ratio || call_ratio >= (1 - min_het_ratio)) {
            i = j;
            continue;
        }
        // this position might be a het candidate, now collect
        // ref calls.
        for (size_t i_read = 0; i_read < ck->reads.n; i_read++) {
            if (ck->reads.a[i_read].start_pos >= pos || ck->reads.a[i_read].end_pos < pos) {
                continue;
            }
            qa_v *readvars = ck->reads.a[i_read].vars;
            int has_alt = 0;
            for (size_t i_var = 0; i_var < readvars->n; i_var++) {
                if (readvars->a[i_var].pos == pos) {
                    has_alt = 1;
                    break;
                }
                if (readvars->a[i_var].pos > pos) {
                    // look back and see if the position is actually covered by a DEL
                    // in which case we should not collect a ref allele.
                    if (i_var > 0) {
                        const uint32_t cigar_i =
                                static_cast<uint32_t>(readvars->a[i_var - 1].allele.n - 1);
                        const uint8_t cigar_op = readvars->a[i_var - 1].allele.a[cigar_i];
                        if (cigar_op == VAR_OP_D) {
                            uint32_t cigar_l = cigar_i;
                            if (readvars->a[i_var - 1].pos + cigar_l >= pos) {
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
                            qa_t *var = &readvars->a[i_var];
                            if (i_var > 0) {
                                qa_t *var_l = &readvars->a[i_var - 1];
                                if (pos - var_l->pos <= var_call_neigh_indel_flank &&
                                    get_cigar_op_qa_t(var_l) & (VAR_OP_D | VAR_OP_I)) {
                                    has_alt = 3;
                                    break;
                                }
                            }
                            if (var->pos - pos <= var_call_neigh_indel_flank &&
                                get_cigar_op_qa_t(var) & (VAR_OP_D | VAR_OP_I)) {
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
                push_to_pg_t(&pg, static_cast<uint32_t>(i_read), pos,
                             static_cast<uint8_t>(VAR_OP_X),
                             static_cast<uint32_t>(readvars->n - 1));
                n_added_ref++;
            }
        }
        i = j;
    }
    if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
        fprintf(stderr, "[dbg::%s] added %d for 2nd piggyback sorting; pg is now length %d\n",
                __func__, n_added_ref, (int)pg.n);
    }

    // note: do not sort qa_v here , piggyback relies on allele as-in-buffer indexing

    // piggyback sorting pileup 2nd pass
    kv_resize(pg_t, pg_buf, pg.n);
    ks_mergesort_kspg(pg.n, pg.a, pg_buf.a);
    kv_destroy(pg_buf);

    // Finding het variants:
    //  1) drop if read varaints' cov<3, or one strand unseen
    //  2) collect total coverage, drop if looking like hom
    //  3) compare alleles, collect the solid ones if any
    for (size_t i = 0; i < pg.n - 1;) {  // loop over positions
        uint32_t pos = pg.a[i].key >> 32;

        size_t j = i + 1;
        for (j = i + 1; j < pg.n; j++) {
            if ((pg.a[j].key >> 32) != pos) {
                break;
            }
        }
        const int cov_tot = static_cast<int>(j) - static_cast<int>(i);
        if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr, "[dbg::%s] pos=%d checkpoint0 ; cov(with ref allele)=%d\n", __func__,
                    (int)pos, cov_tot);
        }

        // collect allele sequences
        // Since in most locations the number of possible alleles should be
        // small, here we just linear search to dedup.
        int pushed_a_var = 0;
        for (size_t k = i; k < i + cov_tot; k++) {
            uint64_t rID = pg.a[k].key & mask_readID;

            if (!is_using_lasm) {
                // for 2a-diploid case, do not count contributions from reads with large clippings
                if (ck->reads.a[rID].left_clip_len > (uint32_t)max_clipping ||
                    ck->reads.a[rID].right_clip_len > (uint32_t)max_clipping) {
                    if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
                        fprintf(stderr,
                                "[dbg::%s]  allelic cov skipped qn=%s (left clip %d, right %d)\n",
                                __func__, ck->qnames[rID]->s, ck->reads.a[rID].left_clip_len,
                                ck->reads.a[rID].right_clip_len);
                    }
                    continue;
                }
            }
            pushed_a_var = 1;

            uint32_t varID = pg.a[k].varID;
            qa_t *read_var = &ck->reads.a[rID].vars->a[varID];
            if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
                fprintf(stderr,
                        "[dbg::%s] collecting allelic coverage: qn=%s var_pos=%d var_len=%d (first "
                        "char %d)\n",
                        __func__, ck->qnames[rID]->s, read_var->pos, (int)read_var->allele.n - 1,
                        read_var->allele.a[0]);
            }
            push_allele_ta_v(ck->varcalls, pos, &read_var->allele, static_cast<uint32_t>(rID));
        }
        if (!pushed_a_var) {
            i = j;  // step forward
            continue;
        }

        ta_t *var = &ck->varcalls->a[ck->varcalls->n - 1];

        // (check allele occurences, and N base occurence)
        if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr, "[dbg::%s] pos=%d checkpoint3 n_alleles=%d; var call min cov is %d\n",
                    __func__, pos, (int)var->alleles.n, var_call_min_cov);
        }
        int n_cov_passed = 0;
        size_t threshold = var_call_min_cov;  // TODO: may need to modify this for multi-allele
        if (var_call_min_cov_ratio > 0) {
            int tmptot = 0;
            for (size_t k = 0; k < var->alleles.n; k++) {
                tmptot += static_cast<int>(var->allele2readIDs.a[k].n);
            }
            uint32_t threshold2 = (uint32_t)(tmptot * var_call_min_cov_ratio);
            threshold = threshold > threshold2 ? threshold : threshold2;
        }

        for (size_t k = 0; k < var->alleles.n; k++) {
            if (var->allele2readIDs.a[k].n >= threshold) {
                var->alleles_is_used.a[k] = 1;
                n_cov_passed++;
            } else {
                var->alleles_is_used.a[k] = 0;
            }
        }
        if (n_cov_passed == (has_ref_allele ? 2 : 1)) {  // 2-allele diploid case
            var->is_used = 1;
            if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
                fprintf(stderr, "[dbg::%s] pos %d accept (sancheck: %d %d\n", __func__, pos,
                        (int)var->allele2readIDs.a[0].n, (int)var->allele2readIDs.a[1].n);
            }
            // Drop any unused alleles now.
            // They are not useful after this point and
            // will complicate variant graph ds.
            cleanup_alleles_ta_t(var);

        } else {  // deallocate the position now and free up the slot
            destroy_ta_t(var, 0, 1);
            var->is_used = 0;
            ck->varcalls->n--;
            if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
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
    log_allele_indices_for_reads_given_varcalls(ck);
    if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
        fprintf(stderr, "[dbg::%s] collected vars on read\n", __func__);
    }

    // update read compatibility using informative sites
    if (need_to_collect_compat) {  // lasm
        update_read_compatibility(ck);
        if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr, "[dbg::%s] updated compat0 (vars)\n", __func__);
        }

        mark_contained_reads(ck, 100);

        if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr, "[dbg::%s] updated compat0 (containment)\n", __func__);
        }
    }

    // cleanup
    kv_destroy(pg);

    return ck;
}

static void vg_get_edge_values1(chunk_t *ck, uint32_t varID1, uint32_t varID2, int counter[4]) {
    for (int i = 0; i < 4; i++) {
        counter[i] = 0;
    }
    assert(varID1 < varID2);
    for (size_t i_read = 0; i_read < ck->reads.n; i_read++) {
        read_t *r = &ck->reads.a[i_read];
        if (r->vars->n > 0) {
            int i1 = -1;
            int i2 = -1;
            for (int i = 0; i < (static_cast<int>(r->vars->n) - 1); i++) {
                assert(r->vars->a[i].is_used);
                if (r->vars->a[i].var_idx == varID1) {
                    i1 = i;
                }
                if (r->vars->a[i].var_idx == varID2) {
                    i2 = i;
                    break;
                }
            }
            if (i1 != -1 && i2 != -1) {
                i1 = r->vars->a[i1].allele_idx;
                i2 = r->vars->a[i2].allele_idx;
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
    radix_sort_ksu32(d, d + 4);
    return d[3] - d[2];
}

static int vg_pos_is_confident(const vg_t *vg, int var_idx) {
    uint32_t tmpcounter[4] = {0, 0, 0, 0};
    for (int tmpi = 0; tmpi < 4; tmpi++) {
        tmpcounter[tmpi] = vg->nodes[var_idx].scores[tmpi];
    }
    int tmpdiff = diff_of_top_two(tmpcounter);
    int ret = tmpdiff > 5;
    return ret;
}

static int vg_init_scores_for_a_location(vg_t *vg, uint32_t var_idx, int do_wipeout) {
    // return 1 when re-init was a hard one
    int ret = -1;
    int debug_print = 0;
    chunk_t *ck = vg->ck;
    uint32_t pos = ck->varcalls->a[var_idx].pos;
    int counter[4] = {0, 0, 0, 0};

    if (var_idx != 0 && !do_wipeout) {
        // reuse values from a previous position
        int i = 0;
        int diff = 5;
        for (int k = var_idx - 1; k > 0; k--) {
            uint32_t pos2 = ck->varcalls->a[k].pos;
            if (debug_print) {
                fprintf(stderr, "trying pos_resume %d (k=%d)\n", pos2, k);
            }
            if (pos - pos2 > 50000) {
                break;  // too far
            }
            if (!vg->next_link_is_broken[k - 1]) {
                if (debug_print) {
                    fprintf(stderr, "trying pos_resume %d (k=%d) checkpoint 1\n", pos2, k);
                }
                if (pos - pos2 > 300 && (pos - pos2 < 5000 ||
                                         i == 0)) {  // search within 5kb or until finally found one
                    uint32_t tmpcounter[4] = {0, 0, 0, 0};
                    for (int tmpi = 0; tmpi < 4; tmpi++) {
                        tmpcounter[tmpi] = vg->nodes[k].scores[tmpi];
                    }
                    int tmpdiff = diff_of_top_two(tmpcounter);
                    if (debug_print) {
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
            if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
                fprintf(stderr, "[dbg::%s] *maybe* re-init pos %d using pos %d\n", __func__,
                        (int)pos, (int)ck->varcalls->a[i].pos);
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
                if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
                    fprintf(stderr,
                            "[dbg::%s] *maybe* not wiping; edge counts are %d %d %d %d; var idx "
                            "are %d and %d\n",
                            __func__, counter[0], counter[1], counter[2], counter[3], i, var_idx);
                }

                uint32_t bests[4];
                vgnode_t *n1 = &vg->nodes[var_idx];
                //for (uint8_t self_combo = 0; self_combo < 4; self_combo++) {
                //    uint32_t score[4];
                //    for (uint8_t prev_combo = 0; prev_combo < 4; prev_combo++) {
                //        int both_hom = ((self_combo == 0 || self_combo == 3) &&
                //                        (prev_combo == 0 || prev_combo == 3));
                //        int s1 = GET_VGN_VAL2(vg, i, prev_combo);
                //        int s2 = counter[(prev_combo >> 1) << 1 | (self_combo >> 1)];
                //        int s3 = both_hom ? 0 : counter[((prev_combo & 1) << 1) | (self_combo & 1)];
                //        score[prev_combo] = s1 + s2 + s3;
                //    }
                //    int source;
                //    bests[self_combo] = max_of_u32_array(score, 4, &source);
                //    n1->scores[self_combo] = bests[self_combo];
                //    n1->scores_source[self_combo] = source;
                //}
                //int best_i;
                //max_of_u32_array(bests, 4, &best_i);
                //n1->best_score_i = best_i;
                for (int tmpi = 0; tmpi < 4; tmpi++) {
                    n1->scores[tmpi] = vg->nodes[i].scores[tmpi];
                    bests[tmpi] = n1->scores[tmpi];
                }
                int best_i;
                max_of_u32_array(bests, 4, &best_i);
                n1->best_score_i = best_i;

                if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
                    for (int tmpi = 0; tmpi < 4; tmpi++) {
                        fprintf(stderr, "[dbg::%s] new %d : %d\n", __func__, tmpi,
                                GET_VGN_VAL2(vg, var_idx, tmpi));
                    }
                }
            }
        } else {
            if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
                fprintf(stderr, "[dbg::%s] tried but failed to find resume point\n", __func__);
            }
            do_wipeout = 1;  // did not find a good resume point, will hard re-init
        }
    }

    // Previously: maybewipe section.
    if (var_idx == 0 || do_wipeout) {
        ret = 1;
        if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr, "[dbg::%s] hard re-init for pos %d\n", __func__, (int)pos);
        }
        for (int i = 0; i < 4; i++) {
            counter[i] = 0;
        }
        for (size_t i_read = 0; i_read < ck->reads.n; i_read++) {
            read_t *r = &ck->reads.a[i_read];
            if (r->start_pos > pos) {
                break;
            }  // reads are loaded from sorted bam, safe to break here
            if (r->end_pos <= pos) {
                continue;
            }
            for (size_t i = 0; i < r->vars->n; i++) {
                if (r->vars->a[i].var_idx == var_idx) {
                    if (r->vars->a[i].allele_idx == 0) {
                        counter[0]++;
                    } else if (r->vars->a[i].allele_idx == 1) {
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

    int best_i;
    uint32_t tmp[4];
    for (int i = 0; i < 4; i++) {
        tmp[i] = GET_VGN_VAL2(vg, var_idx, i);
    }
    max_of_u32_array(tmp, 4, &best_i);
    vg->nodes[var_idx].best_score_i = best_i;
    for (int i = 0; i < 4; i++) {
        vg->nodes[var_idx].scores_source[i] = 4;  // sentinel
    }
    assert(ret >= 0);
    return ret;
}

/*** 2-allele diploid local variant graph ***/
vg_t *vg_gen(chunk_t *ck) {
    if (!ck) {
        return NULL;
    }

    if (ck->varcalls->n == 0) {
        return NULL;
    }

    vg_t *vg = (vg_t *)calloc(1, sizeof(vg_t));
    if (!vg) {
        if (DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr, "[E::%s] calloc failed (vg)\n", __func__);
        }
        return NULL;
    }
    vg->ck = ck;
    vg->n_vars = static_cast<uint32_t>(ck->varcalls->n);

    vg->nodes = (vgnode_t *)calloc(vg->n_vars, sizeof(vgnode_t));
    if (!vg->nodes) {
        if (DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr, "[E::%s] calloc failed (vg nodes)\n", __func__);
        }
        free(vg);
        return NULL;
    }

    vg->edges = (vgedge_t *)calloc(vg->n_vars - 1, sizeof(vgedge_t));
    if (!vg->edges) {
        if (DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr, "[E::%s] calloc failed (vg edges)\n", __func__);
        }
        free(vg->nodes);
        free(vg);
        return NULL;
    }

    vg->next_link_is_broken = (uint8_t *)calloc(vg->n_vars, 1);
    if (!vg->next_link_is_broken) {
        if (DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr, "[E::%s] calloc failed (vg next_link_is_broken)\n", __func__);
        }
        free(vg->edges);
        free(vg->nodes);
        free(vg);
        return NULL;
    }

    // fill in nodes
    for (size_t i = 0; i < ck->varcalls->n; i++) {
        assert(ck->varcalls->a[i].is_used);
        init_vgnode_t(&vg->nodes[i], static_cast<uint32_t>(i));
        vg->nodes[i].del = 0;
    }

    // fill in edges
    for (size_t i_read = 0; i_read < ck->reads.n; i_read++) {
        read_t *r = &ck->reads.a[i_read];
        if (r->vars->n > 0) {
            for (size_t i = 0; i < r->vars->n - 1; i++) {
                assert(r->vars->a[i].is_used);
                uint32_t varID1 = r->vars->a[i].var_idx;
                uint32_t varID2 = r->vars->a[i + 1].var_idx;
                if (varID2 - varID1 != 1) {
                    continue;
                }
                const uint8_t i1 = static_cast<uint8_t>(r->vars->a[i].allele_idx);
                const uint8_t i2 = static_cast<uint8_t>(r->vars->a[i + 1].allele_idx);
                if ((i1 != 0 && i1 != 1) || (i2 != 0 && i2 != 1)) {
                    fprintf(stderr,
                            "[E::%s] this impl is 2-allele diploid, sancheck failed; should not "
                            "happen here, check code. Not incrementing edge weight\n",
                            __func__);
                } else {
                    vg->edges[varID1].counts[i1 << 1 | i2]++;
                }
            }
        }
    }

    // init the first node
    vg_init_scores_for_a_location(vg, 0, 1);

    return vg;
}

static void vg_propogate_one_step(vg_t *vg, int *i_prev_, int i_self) {
    // note: we redo initialization at cov dropouts rather than
    // let backtracing figure out these phasing breakpoints.
    // The breakpoints are stored in vg. When haptagging
    // a read, we will not mix evidences from different phase blocks.
    int debug_print = 0;
    assert(i_self > 0);
    int i_prev = *i_prev_;
    vgnode_t *n1 = &vg->nodes[i_self];

    uint32_t bests[4];
    if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
        vchar_t *a0 = nt4seq2seq(&vg->ck->varcalls->a[i_self].alleles.a[0],
                                 static_cast<int>(vg->ck->varcalls->a[i_self].alleles.a[0].n) - 1);
        vchar_t *a1 = nt4seq2seq(&vg->ck->varcalls->a[i_self].alleles.a[1],
                                 static_cast<int>(vg->ck->varcalls->a[i_self].alleles.a[1].n) - 1);
        fprintf(stderr, "[dbg::%s] i_self=%d (pos=%d a1=%s a2=%s):\n", __func__, i_self,
                vg->ck->varcalls->a[i_self].pos, a0->a, a1->a);
        kv_destroy(*a0);
        kv_destroy(*a1);
        free(a0);
        free(a1);
    }

    // check whether we have a coverage dropout
    for (int i = 0; i < 4; i++) {
        bests[i] = GET_VGE_VAL2(vg, i_prev, i);
    }
    uint32_t best = max_of_u32_array(bests, 4, 0);
    if (best < 3) {  // less than 3 reads support any combination, spot is
                     // a coverage dropout, redo initialization.
        int reinit_failed = vg_init_scores_for_a_location(vg, i_self, 0);
        if (reinit_failed) {
            vg->next_link_is_broken[i_prev] = 1;
            vg->has_breakpoints = 1;
        }
        if (!vg_pos_is_confident(vg, i_self)) {
            vg->next_link_is_broken[i_self] = 1;
        }
        *i_prev_ = i_self;
        if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr, "[dbg::%s]    ! phasing broke at %d (coverage dropout)\n", __func__,
                    vg->ck->varcalls->a[i_self].pos);
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
            if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
                fprintf(stderr,
                        "[dbg::%s]  self combo %d, %d + %d + %d = %d(i_prev=%d; key1=%d key2=%d)\n",
                        __func__, self_combo, s1, s2, s3, score[prev_combo], i_prev,
                        (prev_combo >> 1) << 1 | (self_combo >> 1),
                        (prev_combo & 1) << 1 | (self_combo & 1));
            }
        }
        int source;
        bests[self_combo] = max_of_u32_array(score, 4, &source);
        n1->scores[self_combo] = bests[self_combo];
        n1->scores_source[self_combo] = static_cast<uint8_t>(source);
    }
    int best_i;
    max_of_u32_array(bests, 4, &best_i);
    n1->best_score_i = best_i;

    // another check: if phasing is broken, redo init for self
    if (best_i == 0 || best_i == 3) {  // decision was hom
        int reinit_failed = vg_init_scores_for_a_location(vg, i_self, 0);
        if (reinit_failed) {
            vg->next_link_is_broken[i_prev] = 1;
            vg->has_breakpoints = 1;
        }
        if (!vg_pos_is_confident(vg, i_self)) {
            vg->next_link_is_broken[i_self] = 1;
        }
        if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr, "[dbg::%s]    ! phasing broke at %d (hom decision = %d)\n", __func__,
                    vg->ck->varcalls->a[i_self].pos, best_i);
        }
    }

    *i_prev_ = i_self;
    if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
        fprintf(stderr, "[dbg::%s]    best i: %d (bests: %d %d %d %d)\n", __func__, best_i,
                bests[0], bests[1], bests[2], bests[3]);
    }
}
void vg_propogate(vg_t *vg) {
    // Fill in scores for nodes.
    int i_prev = 0;
    for (uint32_t i = 1; i < vg->n_vars; i++) {
        vg_propogate_one_step(vg, &i_prev, i);
    }
}

void vg_haptag_reads(vg_t *vg) {
    // 2-allele diploid!
    // Assign haptags to reads.
    int debug_print = 0;
    chunk_t *ck = vg->ck;
    int sancheck_cnt[5] = {
            0, 0,
            0,  // no variant
            0,  // ambiguous
            0   // unphasd due to conflict
    };
    uint32_t var_i_start = 0;
    uint32_t var_i_end = vg->n_vars;
    vu64_t buf;
    kv_init(buf);

    for (size_t i_read = 0; i_read < ck->reads.n; i_read++) {
        read_t *r = &ck->reads.a[i_read];
        if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr, "[dbg::%s] saw qn %s\n", __func__, ck->qnames[i_read]->s);
        }

        if (r->vars->n == 0) {
            if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
                fprintf(stderr, "[dbg::%s] skip %s (no var)\n", __func__, ck->qnames[i_read]->s);
            }
            r->hp = HAPTAG_UNPHASED;
            sancheck_cnt[2]++;
            continue;
        }

        if (vg->has_breakpoints) {
            // find the largest phase block overlapping
            // with the read, and use only its variants.
            buf.n = 0;
            uint64_t cnt = 0;
            uint32_t start = r->vars->a[0].var_idx;
            int j = -1;
            int broken = 0;
            for (size_t i = 0; i < r->vars->n; i++) {
                int idx = r->vars->a[i].var_idx;
                if (j == -1) {
                    j = idx;
                }
                while (j < idx + 1) {
                    if (vg->next_link_is_broken[j]) {
                        broken = 1;
                        break;
                    }
                    cnt++;
                    j++;
                }
                if (broken) {
                    kv_push(uint64_t, buf, cnt << 32 | start);
                    start = idx + 1;
                    broken = 0;
                    cnt = 0;
                }
            }
            if (cnt > 0) {
                kv_push(uint64_t, buf, cnt << 32 | start);
            }
            // (do we have any variants?)
            if (buf.n == 0) {
                if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
                    fprintf(stderr, "[dbg::%s] skip %s (no intersecting var)\n", __func__,
                            ck->qnames[i_read]->s);
                }
                r->hp = HAPTAG_UNPHASED;
                sancheck_cnt[2]++;
                continue;
            }

            // (get largest block)
            radix_sort_ksu64(buf.a, buf.a + buf.n);
            var_i_start = (uint32_t)buf.a[buf.n - 1];
            for (uint32_t i = var_i_start; i < vg->n_vars; i++) {
                var_i_end = i + 1;
                if (vg->next_link_is_broken[i]) {
                    break;
                }
            }
            if (var_i_end <= var_i_start) {
                var_i_end = var_i_start + 1;  // interval specified as [)
            }
            if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
                fprintf(stderr,
                        "[dbg::%s] (now using [s=%d e=%d] (%d blocks available; read has %d vars):",
                        __func__, var_i_start, var_i_end, (int)buf.n, (int)r->vars->n);
                for (uint32_t i = var_i_start; i < var_i_end; i++) {
                    fprintf(stderr, "%d, ", vg->ck->varcalls->a[i].pos);
                }
                fprintf(stderr, "\n");
            }
        }

        int votes[2] = {0, 0};
        int veto = 0;
        for (int i = 0; i < static_cast<int>(r->vars->n); i++) {
            const int i_pos = i;
            if (r->vars->a[i].var_idx < var_i_start) {
                continue;
            }
            if (r->vars->a[i].var_idx >= var_i_end) {
                continue;
            }
            if (vg->nodes[r->vars->a[i].var_idx].del) {
                uint32_t pos = vg->ck->varcalls->a[r->vars->a[i].var_idx].pos;
                if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
                    fprintf(stderr, "[dbg::%s] veto at pos=%d\n", __func__, pos);
                }
                veto++;
                continue;
            }
            const uint8_t combo =
                    static_cast<uint8_t>(vg->nodes[r->vars->a[i].var_idx].best_score_i);
            if (combo == 0 || combo == 3) {
                continue;  // position's best phase is a hom
            }
            int idx = r->vars->a[i_pos].allele_idx;
            if (idx == (combo >> 1)) {
                if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
                    fprintf(stderr, "[dbg::%s]    %s pos=%d hap 0 (idx=%d combo=%d)\n", __func__,
                            ck->qnames[i_read]->s, r->vars->a[i_pos].pos, idx, combo);
                }
                votes[0]++;
            } else if (idx == (combo & 1)) {
                if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
                    fprintf(stderr, "[dbg::%s]    %s pos=%d hap 1 (idx=%d combo=%d)\n", __func__,
                            ck->qnames[i_read]->s, r->vars->a[i_pos].pos, idx, combo);
                }
                votes[1]++;
            } else {
                fprintf(stderr,
                        "[E::%s] %s qn=%d impossible (combo=%d idx=%d), check code. This read will "
                        "be untagged.\n",
                        __func__, ck->qnames[i_read]->s, r->vars->a[i_pos].pos, combo, idx);
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

        if (debug_print && DEBUG_LOCAL_HAPLOTAGGING) {
            fprintf(stderr, "[dbg::%s] qname %s vote0=%d vote1=%d veto=%d => hp=%d\n", __func__,
                    ck->qnames[i_read]->s, votes[0], votes[1], veto, r->hp);
        }
    }
    kv_destroy(buf);
    if (DEBUG_LOCAL_HAPLOTAGGING) {
        fprintf(stderr,
                "[M::%s] n_reads %d, hap0=%d hap1=%d no_variant=%d ambiguous=%d "
                "unphased_due_conflict=%d\n",
                __func__, (int)ck->reads.n, sancheck_cnt[0], sancheck_cnt[1], sancheck_cnt[2],
                sancheck_cnt[3], sancheck_cnt[4]);
    }
}

void *kadayashi_local_haptagging_dvr_single_region1(samFile *fp_bam,
                                                    hts_idx_t *fp_bai,
                                                    sam_hdr_t *fp_header,
                                                    faidx_t *fai,
                                                    const char *ref_name,
                                                    uint32_t ref_start,
                                                    uint32_t ref_end,
                                                    int disable_interval_expansion,
                                                    int min_base_quality,
                                                    int min_varcall_coverage,
                                                    float min_varcall_fraction,
                                                    int varcall_indel_mask_flank,
                                                    int max_clipping,
                                                    int max_read_per_region) {
    // Return:
    //    Unless hashtable's initialization or input opening failed,
    //    this function always return a hashtable, which might be empty
    //    if no read can be tagged or there was any error
    //    when tagging reads.

    bamfile_t *hf = init_bamfile_t_with_opened_files(fp_bam, fp_bai, fp_header);
    if (!hf) {
        return NULL;
    }

    kh_str2int_t *qname2tag = (kh_str2int_t *)khash_str2int_init();
    if (!qname2tag) {
        return NULL;
    }

    chunk_t *ck = unphased_varcall_a_chunk(
            hf, 0, fai, ref_name, ref_start, ref_end, disable_interval_expansion, min_base_quality,
            min_varcall_coverage, min_varcall_fraction, varcall_indel_mask_flank,
            0,  // use vcf input?
            0,  // use lasm?
            max_clipping, max_read_per_region);

    vg_t *vg = 0;
    vg = vg_gen(ck);
    if (vg) {
        vg_propogate(vg);
        vg_haptag_reads(vg);
        int absent;
        for (size_t i = 0; i < ck->reads.n; i++) {
            int haptag = ck->reads.a[i].hp + 1;  // use 1-index
            char *qn = ck->qnames[i]->s;
            const int qn_l = static_cast<int>(ck->qnames[i]->l);
            const size_t qname_key_len = qn_l + 1;
            char *qname_key = (char *)calloc(qname_key_len, 1);
            if (!qname_key) {
                if (DEBUG_LOCAL_HAPLOTAGGING) {
                    fprintf(stderr, "[E::%s] calloc failed\n", __func__);
                }
                destroy_vg_t(vg);
                if (ck) {
                    destroy_chunk_t(ck, 1);
                }
                destroy_holder_bamfile_t(hf, 1);
                return nullptr;
            }
            snprintf(qname_key, qname_key_len, "%s", qn);
            kh_put(str2int, qname2tag, qname_key, &absent);
            if (absent) {
                khash_str2int_set(qname2tag, qname_key, haptag);
            } else {
                free(qname_key);
            }
        }
        destroy_vg_t(vg);
    }
    if (ck) {
        destroy_chunk_t(ck, 1);
    }

    destroy_holder_bamfile_t(hf, 1);
    return qname2tag;
}

std::unordered_map<std::string, int32_t> kadayashi_dvr_single_region_wrapper(
        samFile *fp_bam,
        hts_idx_t *fp_bai,
        sam_hdr_t *fp_header,
        faidx_t *fai,
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
    std::shared_ptr<kh_str2int_t> qname2tag(
            (kh_str2int_t *)kadayashi::kadayashi_local_haptagging_dvr_single_region1(
                    fp_bam, fp_bai, fp_header, fai, ref_name.c_str(), ref_start, ref_end,
                    disable_interval_expansion, min_base_quality, min_varcall_coverage,
                    min_varcall_fraction, varcall_indel_mask_flank, max_clipping,
                    max_read_per_region),
            [](kh_str2int_t *ptr) { khash_str2int_destroy_free((void *)ptr); });

    if (!qname2tag) {
        throw std::runtime_error("Kadayashi returned a nullptr!");
    }

    // Convert the C-style hash to C++.
    std::unordered_map<std::string, int32_t> ret;
    for (khint_t i = 0; i < kh_end(qname2tag.get()); ++i) {
        if (kh_exist(qname2tag.get(), i)) {
            const std::string key = kh_key(qname2tag.get(), i);
            const int32_t val = kh_val(qname2tag.get(), i);
            ret[key] = val;
        }
    }

    return ret;
}

}  // namespace kadayashi
