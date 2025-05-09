#ifndef KADAYASHI_TYPES_H
#define KADAYASHI_TYPES_H

// clang-format off
#include "kvec.h"

#include <htslib/khash.h>
#include <htslib/khash_str2int.h>

#include <assert.h>
// clang-format on

struct hts_idx_t;
struct bamfile_t;
struct hts_base_mod_state;
struct bam1_t;
struct sam_hdr_t;
typedef struct htsFile samFile;
typedef struct sam_hdr_t bam_hdr_t;

#define KDYS_DISABLE_MAX_READ_CAP (1 << 27)
#define KDYS_DISABLE_REGION_EXPANSION 1

namespace kadayashi {

#define VAR_OP_M 1
#define VAR_OP_X 2
#define VAR_OP_I 4
#define VAR_OP_D 8
#define HAPTAG_UNPHASED 254
#define HAPTAG_AMBPHASED 99

#define CMPT0_COMPAT 1
#define CMPT0_INCOMPAT (1 << 1)
#define CMPT0_AWAY (1 << 2)  // aligned ranges do not overlap
#define CMPT0_CONTAINED (1 << 7)
#define CMPT0_DEL (1 << 6)
#define CMPT0_F (CMPT0_INCOMPAT | CMPT0_CONTAINED | CMPT0_DEL)

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

#define READLINE_BUF_LEN 1024

// clang-format off
static const unsigned char kdy_seq_nt4_table[256] = {
        // translate ACG{T,U} to 0123 case insensitive
        4, 4,       4, 4,       4,       4,       4, 4,       4, 4, 4, 4, 4, 4,       4,       4,
        4, 4,       4, 4,       4,       4,       4, 4,       4, 4, 4, 4, 4, 4,       4,       4,
        4, 4,       4, 4,       4,       4,       4, 4,       4, 4, 4, 4, 4, 4,       4,       4,
        4, 4,       4, 4,       4,       4,       4, 4,       4, 4, 4, 4, 4, 4,       4,       4,
        4, 0 /*A*/, 4, 1 /*C*/, 4,       4,       4, 2 /*G*/, 4, 4, 4, 4, 4, 5 /*M*/, 4 /*N*/, 4,
        4, 4,       4, 4,       3 /*T*/, 3 /*U*/, 4, 4,       4, 4, 4, 4, 4, 4,       4,       4,
        4, 0 /*a*/, 4, 1 /*c*/, 4,       4,       4, 2 /*g*/, 4, 4, 4, 4, 4, 5 /*m*/, 4 /*n*/, 4,
        4, 4,       4, 4,       3 /*t*/, 3 /*u*/, 4, 4,       4, 4, 4, 4, 4, 4,       4,       4,
        4, 4,       4, 4,       4,       4,       4, 4,       4, 4, 4, 4, 4, 4,       4,       4,
        4, 4,       4, 4,       4,       4,       4, 4,       4, 4, 4, 4, 4, 4,       4,       4,
        4, 4,       4, 4,       4,       4,       4, 4,       4, 4, 4, 4, 4, 4,       4,       4,
        4, 4,       4, 4,       4,       4,       4, 4,       4, 4, 4, 4, 4, 4,       4,       4,
        4, 4,       4, 4,       4,       4,       4, 4,       4, 4, 4, 4, 4, 4,       4,       4,
        4, 4,       4, 4,       4,       4,       4, 4,       4, 4, 4, 4, 4, 4,       4,       4,
        4, 4,       4, 4,       4,       4,       4, 4,       4, 4, 4, 4, 4, 4,       4,       4,
        4, 4,       4, 4,       4,       4,       4, 4,       4, 4, 4, 4, 4, 4,       4,       4,
};
// clang-format on

typedef kvec_t(uint8_t) vu8_t;
typedef kvec_t(uint64_t) vu64_t;
typedef kvec_t(uint32_t) vu32_t;
typedef kvec_t(uint16_t) vu16_t;
typedef kvec_t(int) vi_t;
typedef kvec_t(float) vfloat_t;
typedef kvec_t(char) vchar_t;
typedef kvec_t(vu8_t) vu8_v;
typedef kvec_t(vu16_t) vu16_v;
typedef kvec_t(vu32_t) vu32_v;
typedef kvec_t(kstring_t *) kstring_v;

// hashtables
KHASH_MAP_INIT_STR(htstri_t, int)
KHASH_MAP_INIT_INT(htu32_t, uint32_t)
KHASH_MAP_INIT_INT(htu64_t, uint64_t)

typedef struct {
    char *fn;
    samFile *fp;
    hts_idx_t *bai;
    bam_hdr_t *header;
    bam1_t *aln;
    hts_base_mod_state *mod;
} bamfile_t;

typedef struct {
    uint32_t pos;
    vu8_t allele;         // as in kdy_seq_nt4_table;
                          // stores alt for SNP and INS,
                          // stores ref for DEL
    uint32_t var_idx;     // as in ta_v buffer
    uint32_t allele_idx;  // as in alleles of ta_v buffer; not collected until
                          // unphased varcall is done
    uint8_t is_used;
    uint8_t hp;
} qa_t;  // query allele

typedef kvec_t(qa_t) qa_v;

typedef struct {
    uint32_t start_pos, end_pos;
    uint32_t ID;  // ID is local to a chunk
    qa_v *vars;   // variants owned by read
    int hp;       // stores the haplotagging result
    int votes_diploid[2];
    uint8_t strand;
    float de;  // gap-compressed seq div
    uint32_t left_clip_len, right_clip_len;
} read_t;
typedef kvec_t(read_t) reads_v;

typedef struct {
    uint32_t pos;
    vu8_v alleles;          // sequences, use 0123 (as in kdy_seq_nt4_table)
    vu8_t alleles_is_used;  // records whether each of the allele has enough
                            // coverage.
    vu32_v allele2readIDs;  // inverse index: in what reads did we see
                            // a given allele at pos. Needed for
                            // variant graph construction.
    uint8_t is_used;        // whether this position is used
} ta_t;                     // possible alleles at a reference position
typedef kvec_t(ta_t) ta_v;

typedef struct {
    reads_v reads;
    ta_v *varcalls;
    kstring_t **qnames;
    khash_t(htstri_t) * qname2ID;
    uint32_t abs_start, abs_end;
    uint8_t **compat0;  // all-v-all read compatibility derived from
                        // the original read-ref alignment, i.e. with
                        // all variants rather than using informative sites.
} chunk_t;

typedef struct {
    char *chrom;
    int bucket_l;
    vu32_t poss;
    vi_t start_indices;
} ref_vars_t;

typedef struct {
    uint32_t counts[4];  // 2-allele diploid,
                         //   <i>[hap0] - <i+1>[hap0]
                         //   <i>[hap0] - <i+1>[hap1]
                         //   <i>[hap1] - <i+1>[hap0]
                         //   <i>[hap1] - <i+1>[hap1]
} vgedge_t;              // connection of position i => position i+1
#define GET_VGE_VAL(vg, i, H1, H2) (vg->edges[(i)].counts[(H1) << 1 | (H2)])
#define GET_VGE_VAL2(vg, i, comb) (vg->edges[(i)].counts[(comb)])

typedef struct {
    uint32_t ID;         // self's index
    uint32_t scores[4];  // 2-allele diploid. 00, 01, 10, 11
    uint8_t scores_source[4];
    uint8_t del;       // allow marking node as not-in-use when
                       // propogating through the graph.
    int best_score_i;  // note: backtracing does not need to know about
                       // the previous node.
} vgnode_t;            // one position; 2-allele diploid
#define GET_VGN_VAL(vg, i, A1, A2) (vg->nodes[(i)].scores[(A1) << 1 | (A2)])
#define GET_VGN_VAL2(vg, i, comb) (vg->nodes[(i)].scores[(comb)])

typedef struct {
    chunk_t *ck;
    uint32_t n_vars;
    vgedge_t *edges;
    vgnode_t *nodes;
    uint8_t *next_link_is_broken;  // logs phasing breakpoints
    uint8_t has_breakpoints;
} vg_t;  // 2-allele diploid!

void destroy_vu32_v(vu32_v *h, int include_self);

ref_vars_t *init_ref_vars_t(const char *chrom, int bucket_l);

void destroy_ref_vars_t(ref_vars_t *h);

void destroy_vg_t(vg_t *vg);

void init_vgnode_t(vgnode_t *h, uint32_t ID);

void destroy_chunk_t(chunk_t *h, int include_self);

void init_read_t(read_t *h);

void destroy_read_t(read_t *h, int include_self);

qa_v *init_qa_v(void);

void destroy_qa_v(qa_v *h);

void destroy_ta_t(ta_t *h, int include_self, int forced);

void destroy_ta_v(ta_v *h);

int sort_qa_v(qa_v *h, qa_v *buf);

void dummyexpand_qa_v(qa_v *h, int n);

void sort_ksqa(qa_t *a, size_t n);
void radix_sort_ksu32(uint32_t *a, uint32_t *b);
void sort_kssu32(uint32_t *a, size_t n);
void radix_sort_ksu64(uint64_t *a, uint64_t *b);
void sort_kssu64(uint64_t *a, size_t n);

}  // namespace kadayashi

#endif  //KADAYASHI_TYPES_H
