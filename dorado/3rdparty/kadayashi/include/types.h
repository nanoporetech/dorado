#ifndef KADAYASHI_TYPES_H
#define KADAYASHI_TYPES_H

#include "hts_types.h"

#include <cassert>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#define KDYS_DISABLE_MAX_READ_CAP (1 << 27)
#define KDYS_DISABLE_REGION_EXPANSION 1

namespace kadayashi {

#define VAR_OP_INVALID 0
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

struct bamfile_t {
    samFile *fp;
    hts_idx_t *bai;
    bam_hdr_t *header;
};

struct qa_t {
    uint32_t pos;
    std::vector<uint8_t> allele;  // as in kdy_seq_nt4_table;
                                  // stores alt for SNP and INS,
                                  // stores ref for DEL
    uint32_t var_idx;             // as in ta_v buffer
    uint32_t allele_idx;          // as in alleles of ta_v buffer; not collected until
                                  // unphased varcall is done
    uint8_t is_used;
    uint8_t hp;
};  // query allele

struct read_t {
    uint32_t start_pos = std::numeric_limits<uint32_t>::max();
    uint32_t end_pos = std::numeric_limits<uint32_t>::max();
    uint32_t ID = std::numeric_limits<uint32_t>::max();  // ID is local to a chunk
    std::vector<qa_t> vars;                              // variants owned by read
    int hp = HAPTAG_UNPHASED;                            // stores the haplotagging result
    int votes_diploid[2] = {0, 0};
    uint8_t strand = std::numeric_limits<uint8_t>::max();
    float de = {0.0f};  // gap-compressed seq div
    uint32_t left_clip_len = {0};
    uint32_t right_clip_len = {0};
};

struct ta_t {
    uint32_t pos;
    std::vector<std::vector<uint8_t>> alleles;  // Sequences, use 0123 (as in kdy_seq_nt4_table)
    std::vector<uint8_t> alleles_is_used;       // Records whether each of the allele has enough
                                                // coverage.
    std::vector<std::vector<uint32_t>> allele2readIDs;  // Inverse index: in what reads did we see
                                                        // a given allele at pos. Needed for
                                                        // variant graph construction.
    uint8_t is_used;                                    // Whether this position is used
};  // possible alleles at a reference position

struct ref_vars_t {
    int bucket_l;
    std::vector<uint32_t> poss;
    std::vector<int> start_indices;
};

struct vgedge_t {
    uint32_t counts[4];  // 2-allele diploid,
                         //   <i>[hap0] - <i+1>[hap0]
                         //   <i>[hap0] - <i+1>[hap1]
                         //   <i>[hap1] - <i+1>[hap0]
                         //   <i>[hap1] - <i+1>[hap1]
};  // connection of position i => position i+1
#define GET_VGE_VAL(vg, i, H1, H2) ((vg).edges[(i)].counts[(H1) << 1 | (H2)])
#define GET_VGE_VAL2(vg, i, comb) ((vg).edges[(i)].counts[(comb)])

struct vgnode_t {
    uint32_t ID{0};                  // self's index
    uint32_t scores[4]{0, 0, 0, 0};  // 2-allele diploid. 00, 01, 10, 11
    uint8_t scores_source[4]{0, 0, 0, 0};
    uint8_t del{0};        // allow marking node as not-in-use when
                           // propogating through the graph.
    int best_score_i{-1};  // note: backtracing does not need to know about
                           // the previous node.
};  // one position; 2-allele diploid
#define GET_VGN_VAL(vg, i, A1, A2) ((vg).nodes[(i)].scores[(A1) << 1 | (A2)])
#define GET_VGN_VAL2(vg, i, comb) ((vg).nodes[(i)].scores[(comb)])

struct vg_t {
    uint32_t n_vars{0};
    std::vector<vgedge_t> edges{};
    std::vector<vgnode_t> nodes{};
    std::vector<uint8_t> next_link_is_broken{};  // logs phasing breakpoints
    uint8_t has_breakpoints{0};
};  // 2-allele diploid graph used by dvr method

struct chunk_t {
    int is_valid;
    std::vector<read_t> reads;
    std::vector<ta_t> varcalls;
    std::vector<std::string> qnames;
    std::unordered_map<std::string, int> qname2ID;
    uint32_t abs_start;
    uint32_t abs_end;
    vg_t vg;
};

struct phase_return_t {
    std::unordered_map<std::string, int> qname2hp = {};
    chunk_t ck = {};
};

struct pileup_pars_t {
    bool disable_region_expansion = false;
    bool allow_any_candidate =
            false;  // we only want this to be true when input has a trusted vcf; otherwise, we want to filter by base quality, coverage, etc
    int min_base_quality = 5;
    int min_varcall_coverage = 5;
    float min_varcall_fraction = 0.2f;
    int varcall_indel_mask_flank = 10;
    int max_clipping = 200;
    uint64_t max_read_per_region = 1ULL << 27;
};

}  // namespace kadayashi

#endif  //KADAYASHI_TYPES_H
