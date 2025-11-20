#pragma once

#include "bam_file_view.h"
#include "types.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct faidx_t;
struct bam1_t;

namespace kadayashi {
// clang-format off

/**
 * @brief Construct a htslib-style region string (1-index, inclusive–inclusive)
 *        from 0-index, inclusive-exclusive coordinates.
 *
 * @param ref_name Reference sequence name.
 * @param start    0-based inclusive start position.
 * @param end      0-based exclusive end position.
 * @return Region string in the format "ref_name:start-end".
 */
std::string create_region_string(const std::string_view ref_name,
                                        const uint32_t start,
                                        const uint32_t end);


/**
 * @brief  Make a hashtable that maps read names to read haptags
 *         given the chunk.
 * @param ck   The chunk's full pileup info.
 */
std::unordered_map<std::string, int> kadayashi_local_haptagging_gen_ht(chunk_t &ck);


/**
 * @brief Perform phasing for a query region with the deepvariant replica method, 
 *        return all info (read haptags, the chunk, and phasing breakpoints).
 * 
 * This function is exposed for CLI use. To obtain only the read haptags, 
 * use `kadayashi_dvr_single_region_wrapper` instead.
 * 
 * @par Thread safety 
 * Seeking the BAM file from other threads with htslib *itr_query methods
 * will produce unpredictable behavior.
 * 
 * @param fp_bam    BAM file. Must present.
 * @param fp_bai    BAM index file. Must present.
 * @param fp_header BAM header. Must present.
 * @param fai       The reference genome. Must present if `pp` does not set
 *                  `disable_low_complexity_masking` to true.
 * @param ref_name  The reference sequence's name.
 * @param ref_start Query interval's start position on the reference sequence.
 *                  0-index, inclusive.
 * @param ref_end   Query interval's end position on the reference sequence.
 *                  0-index, exclusive.
 * @param pp        Parameters used for filtering variants during pileup.
 */
phase_return_t kadayashi_local_haptagging_dvr_single_region(samFile *fp_bam,
                                                            hts_idx_t *fp_bai,
                                                            sam_hdr_t *fp_header,
                                                            const faidx_t *fai,
                                                            const std::string_view ref_name,
                                                            const uint32_t ref_start,
                                                            const uint32_t ref_end,
                                                            const pileup_pars_t &pp);


/**
 * @brief Perform phasing for a query region with the simple phasing method (flip-flop), 
 *        return all info (read haptags, the chunk, and phasing breakpoints).
 * 
 * Like `kadayashi_local_haptagging_dvr_single_region`, this function 
 * is exposed for CLI use. To obtain only the read haptags, use 
 * `kadayashi_simple_single_region_wrapper` instead.
 * 
 * @par Thread safety 
 * Seeking the BAM file from other threads with htslib *itr_query methods
 * will produce unpredictable behavior.
 * 
 * @param fp_bam    BAM file. Must present.
 * @param fp_bai    BAM index file. Must present.
 * @param fp_header BAM header. Must present.
 * @param fai       The reference genome. Must present if `pp` does not set
 *                  `disable_low_complexity_masking` to true.
 * @param ref_name  The reference sequence's name.
 * @param ref_start Query interval's start position on the reference sequence.
 *                  0-index, inclusive.
 * @param ref_end   Query interval's end position on the reference sequence.
 *                  0-index, exclusive.
 * @param pp        Parameters used for filtering variants during pileup.
 */
phase_return_t kadayashi_local_haptagging_simple_single_region(samFile *fp_bam,
                                                               hts_idx_t *fp_bai,
                                                               sam_hdr_t *fp_header,
                                                               const faidx_t *fai,
                                                               const std::string_view ref_name,
                                                               const uint32_t ref_start,
                                                               const uint32_t ref_end,
                                                               const pileup_pars_t &pp);


/// Used to store read depth info of a variant candidate allele.
struct cov_t {
    int cov_hap0 = 0;
    int cov_hap1 = 0;
    int cov_unphased = 0;

    /// for temporary usage
    int cov_tot = 0;  

    /// forward stand coverage (regardless of phase)
    int cov_fwd = 0;  
    /// reverse stand coverage (regardless of phase)
    int cov_bwd = 0;  
};


/// A varaint candidate allele.
struct vc_allele_t {
    /// read depth
    cov_t cov;

    /// the allele's sequence in u8i, where the last slot is cigar op.
    std::vector<uint8_t> allele = {};  
};


constexpr uint8_t FLAG_VAR_NA      = 0;
constexpr uint8_t FLAG_VAR_HOM     = 1;
constexpr uint8_t FLAG_VAR_HET     = 2;
constexpr uint8_t FLAG_VAR_MULTHET = 4;

constexpr uint8_t FLAG_VARSTAT_MAYBE    = 0;  // temporary state
constexpr uint8_t FLAG_VARSTAT_ACCEPTED = 1;
constexpr uint8_t FLAG_VARSTAT_REJECTED = 2;
constexpr uint8_t FLAG_VARSTAT_UNSURE   = 4;
constexpr uint8_t FLAG_VARSTAT_UNKNOWN  = 8;

struct vc_variants1_val_t{
     uint8_t is_accepted = 0;
     uint8_t type = 0;
     std::vector<vc_allele_t> alleles = {};
};

/// variants on one reference or interval
typedef std::unordered_map<uint32_t, vc_variants1_val_t> vc_variants1_t;  

/// variants on multiple references
typedef std::unordered_map<std::string, vc_variants1_t> vc_variants_t;  


/**
 * @brief Parse read pile up to produce variant candidates. 
 * 
 * Exposed for CLI. To perform read phasing and varcall, use the wrappers.
 * This routine is used for both phasing and pileup-based variant calling.
 * For the former, set `qname2hp` to null and set `pp` to have 
 *  the pileup only report het variants. 
 * For the latter, supply `qname2hp` and set appropriate values for `pp`.
 * 
 * @par Thread safety
 * BAM file should be thread local because seeking it from other 
 * threads may be unsafe.
 * 
 * @param hf         BAM file, index and header.
 * @param ht_refvars If not empty, the pileup will only report variants
 *                   that exist in this list. Only use this when caller
 *                   has an authoritative superset of variants.
 * @param fai        The reference genome.
 * @param qname2hp   Read haptags. 0-index.
 * @param refname    The reference sequence's name.
 * @param itvl_start The query interval's start position on reference.
 *                   0-index, inclusive.
 * @param itvl_end   The query interval's end position on reference.
 *                   0-index, exclusive.
 * @param pp         Parameters for the pileup and variant filtering.
 */
chunk_t variant_pileup_ht(BamFileView &hf,
                       const variants_t &ht_refvars,
                       const faidx_t *fai,
                       const str2int_t *qname2hp,
                       const std::string_view refname,
                       const uint32_t itvl_start,
                       const uint32_t itvl_end,
                       const pileup_pars_t &pp);



/**
 * @brief Phase reads in a query region. Produces a hashtable mapping
 *        from read names to their haptags in 0-index.

 * @par Thread safety
 * BAM file.
 * 
 * @param fp_bam     BAM file. Must present.
 * @param fp_bai     BAM index. Must present.
 * @param fp_header  BAM header. Must present.
 * @param fai        Reference genome.
 * @param ref_name   Reference sequence name.
 * @param ref_start  Start of query region, 0-index inclusive.
 * @param ref_end    End of query region, 0-index exclusive.
 * @param disable_interval_expansion If set, the phasing will only use  
 *                   variants that are inside the query region. This might
 *                   not be desirable when the query happens to start in a
 *                   long homozygous region and could've been phased with 
 *                   variant(s) right outside of the query. 
 *                   If not set, phasing will consider variants to the left 
 *                   and the right of the query region, up to 50kb away.
 * @param min_base_quality Filter out variation on read if any base in it
 *                   has base quality lower than this value.
 *                   Has no effect for deletions. 
 * @param min_varcall_coverage Candidate variant's alt allele read depth
 *                   must at least be this value.
 * @param min_varcall_fraction Candidate variant's alt allele frequency
 *                   must at least be this value.
 * @param max_clipping Filter out reads with clippings at least this long.
 * @param max_gapcompressed_seqdiv If `de:f` tag exists in BAM records, 
 *                     filter out reads with values at least this large.
 */
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
        const float max_gapcompressed_seqdiv);


/** Same as `kadayashi_dvr_single_region_wrapper` except for that 
 * phasing is performed with the simple phasing method (flip-flop) 
 * instead of deepvariant replica.
 */ 
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
        const float max_gapcompressed_seqdiv);


/// Exposed for CLI. Used `variant_dorado_style_t` instead.
struct variant_fullinfo_t{
    bool is_valid{true};
    bool is_confident{true};
    bool is_multi_allele{false};

    uint32_t pos0{0};  // 0-index
    uint8_t qual0{60};
    std::string ref_allele_seq0{};
    std::string alt_allele_seq0{};
    bool is_phased0{false};
    char genotype0[3]={'0','/', '0'};  

    uint32_t pos1{0};  // 0-index
    uint8_t qual1{60};
    std::string ref_allele_seq1{};
    std::string alt_allele_seq1{};
    bool is_phased1{false};
    char genotype1[3]={'0', '/', '0'};  
};


/** Variant information that can be directly used to compose 
* VCF line, except for `pos` which is in 0-index.
*/
struct variant_dorado_style_t{
    bool is_confident;
    bool is_phased;
    uint32_t pos;
    
    /** Currently a placeholder value: confident varaints 
     * are assigned 60. Unsure variants are assigned 0.
     */ 
    int qual;

    std::string ref;
    std::vector<std::string> alts;
    std::pair<char, char> genotype;
};

bool operator==(const variant_fullinfo_t &a, const variant_fullinfo_t &b);
bool operator==(const variant_dorado_style_t &a, const variant_dorado_style_t &b);
struct varcall_result_internal_t {
    std::unordered_map<std::string, int> qname2hp{};  // 0-index
    std::vector<variant_fullinfo_t> variants{};
    std::unordered_map<uint32_t, uint8_t> phasing_breakpoints{};
};

struct varcall_result_t {
    std::unordered_map<std::string, int> qname2hp{};  // 0-index
    std::vector<variant_dorado_style_t> variants{};   // 0-index, for VCF
    std::unordered_map<uint32_t, uint8_t> phasing_breakpoints{};
};

struct ck_and_varcall_result_t {
    chunk_t ck{};
    varcall_result_internal_t vr{};
};


/// Exposed for CLI usage. Use `kadayashi_phase_and_varcall_wrapper` instead.
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
                                                    const bool use_dvr_for_phasing);

/**
 * @brief Phase reads in a query region and perform phased variant calling. 

 * @par Thread safety
 * BAM file.
 * 
 * @param fp_bam     BAM file. Must present.
 * @param fp_bai     BAM index. Must present.
 * @param fp_header  BAM header. Must present.
 * @param fai        Reference genome.
 * @param ref_name   Reference sequence name.
 * @param ref_start  Start of query region, 0-index inclusive.
 * @param ref_end    End of query region, 0-index exclusive.
 * @param disable_interval_expansion If set, the phasing will only use  
 *                   variants that are inside the query region. This might
 *                   not be desirable when the query happens to start in a
 *                   long homozygous region and could've been phased with 
 *                   variant(s) right outside of the query. 
 *                   If not set, phasing will consider variants to the left 
 *                   and the right of the query region, up to 50kb away.
 * @param min_base_quality Filter out variation on read if any base in it
 *                   has base quality lower than this value.
 *                   Has no effect for deletions. 
 * @param min_varcall_coverage Candidate variant's alt allele read depth
 *                   must at least be this value.
 * @param min_varcall_fraction Candidate variant's alt allele frequency
 *                   must at least be this value.
 * @param max_clipping Filter out reads with clippings at least this long.
 * @param max_gapcompressed_seqdiv If `de:f` tag exists in BAM records, 
 *                     filter out reads with values at least this large.
 * @param use_dvr_for_phasing If set, use deepvariant replica phasing method.
 *                            Otherwise, the simple phasing method (flip-flop)
 *                            will be used.
 */
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
                                                     const bool use_dvr_for_phasing);

// clang-format on
}  // namespace kadayashi
