#pragma once

#include "hts_types.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace kadayashi {

constexpr bool DEBUG_LOCAL_HAPLOTAGGING = false;
constexpr uint8_t VAR_OP_INVALID = 0;
constexpr uint8_t VAR_OP_M = 1;
constexpr uint8_t VAR_OP_X = 2;
constexpr uint8_t VAR_OP_I = 4;
constexpr uint8_t VAR_OP_D = 8;
constexpr uint8_t HAPTAG_UNPHASED = 254;
constexpr uint8_t HAPTAG_AMBPHASED = 99;

// clang-format off
constexpr uint8_t SENTINEL_REF_ALLELE_INT = 5;  // kdy_seq_nt4_table
constexpr unsigned char kdy_seq_nt4_table[256] = {
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
inline bool operator<(const qa_t& a, const qa_t& b) { return a.pos < b.pos; }

struct read_t {
    uint32_t start_pos{std::numeric_limits<uint32_t>::max()};
    uint32_t end_pos{std::numeric_limits<uint32_t>::max()};
    uint32_t ID{std::numeric_limits<uint32_t>::max()};  // ID is local to a chunk
    std::vector<qa_t> vars{};                           // variants owned by read
    uint8_t hp{HAPTAG_UNPHASED};                        // stores the haplotagging result
    int votes_diploid[2]{0, 0};
    uint8_t strand{std::numeric_limits<uint8_t>::max()};
    float de{0.0f};  // gap-compressed seq div
    int left_clip_len{0};
    int right_clip_len{0};
};

constexpr uint8_t TA_STAT_UNCALLED = 0;
constexpr uint8_t TA_STAT_ACCEPTED = 1;
constexpr uint8_t TA_STAT_UNSURE = 2;
constexpr uint8_t TA_TYPE_HOM = 0;
constexpr uint8_t TA_TYPE_HET = 1;
constexpr uint8_t TA_TYPE_HETMULTI = 2;
constexpr uint8_t TA_TYPE_UNKNOWN = 4;
struct ta_t {
    uint32_t pos;
    std::vector<std::vector<uint8_t>> alleles;  // Sequences, use 0123 (as in kdy_seq_nt4_table)
    std::vector<uint8_t> alleles_is_used;       // Records whether each of the allele has enough
                                                // coverage.
    std::vector<std::vector<uint32_t>> allele2readIDs;  // Inverse index: in what reads did we see
                                                        // a given allele at pos. Needed for
                                                        // variant graph construction.
    uint8_t is_used : 4, type : 4;
    std::array<char, 4> genotype{'.', '/', '.', '\0'};
};  // possible alleles at a reference position

struct variant_t {
    uint8_t op;              // VAR_OP_{M,X,I,D} as in cigar operation
    std::string alt_allele;  // not storing ref allele, we assume that
                             // we will have access to the original vcf or its data
                             // when writing the output vcf.
};  // for global phasing; variant is load from file or memory, and will not change
typedef std::unordered_map<uint32_t, variant_t> variants_t;
typedef std::unordered_map<std::string, variants_t> reference_variants_t;

typedef std::unordered_map<uint32_t, uint8_t> var2hap_t;  // haptag of reference alllel
typedef std::unordered_map<std::string, var2hap_t> varhaps_t;

typedef std::unordered_map<std::string, int> str2int_t;

struct vgedge_t {
    uint32_t counts[4];  // 2-allele diploid,
                         //   <i>[hap0] - <i+1>[hap0]
                         //   <i>[hap0] - <i+1>[hap1]
                         //   <i>[hap1] - <i+1>[hap0]
                         //   <i>[hap1] - <i+1>[hap1]
};  // connection of position i => position i+1

struct vgnode_t {
    uint32_t ID{0};                  // self's index
    uint32_t scores[4]{0, 0, 0, 0};  // 2-allele diploid. 00, 01, 10, 11
    uint8_t scores_source[4]{0, 0, 0, 0};
    uint8_t del{0};        // allow marking node as not-in-use when
                           // propogating through the graph.
    int best_score_i{-1};  // note: backtracing does not need to know about
                           // the previous node.
};  // one position; 2-allele diploid

struct variant_graph_t {
    uint32_t n_vars{0};
    std::vector<vgedge_t> edges{};
    std::vector<vgnode_t> nodes{};
    std::vector<uint8_t> next_link_is_broken{};  // logs phasing breakpoints
    uint8_t has_breakpoints{0};
};  // 2-allele diploid graph used by dvr method

struct chunk_t {
    bool is_valid;
    std::vector<read_t> reads;
    std::vector<ta_t> varcalls;
    std::vector<std::string> qnames;
    std::unordered_map<std::string, int> qname2ID;
    uint32_t abs_start;
    uint32_t abs_end;
    std::string refname;
    variant_graph_t vg;
};

struct phase_return_t {
    std::unordered_map<std::string, int> qname2hp{};
    chunk_t ck{};
    std::unordered_map<uint32_t, uint8_t> phasing_breakpoints{};
};

struct pileup_pars_t {
    bool allow_any_candidate{
            false};  // don't use; only set this for experimenting with global phasing
    int min_base_quality{5};
    int min_varcall_coverage{5};
    float min_varcall_fraction{0.2f};
    int max_clipping{200};
    int min_mapq{10};
    int min_strand_cov{1};
    float min_strand_cov_frac{0.033f};

    float max_gapcompressed_seqdiv{0.1f};

    bool retain_het_only{true};
    bool retain_SNP_only{true};
    bool use_bloomfilter{false};
    bool disable_low_complexity_masking{false};
    bool disable_region_expansion{false};
};

typedef std::unordered_map<std::string, std::vector<std::pair<uint32_t, uint32_t>>> intervals_t;
struct query_region_t {
    std::string chrom{};
    uint32_t start{0};
    uint32_t end{0};
};
typedef std::unordered_map<std::string, std::vector<query_region_t>> query_regions_t;

}  // namespace kadayashi
