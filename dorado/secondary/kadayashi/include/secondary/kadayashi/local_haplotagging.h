#pragma once

#include "secondary/common/bam_file.h"
#include "secondary/common/variant.h"
#include "secondary/features/medaka_read_matrix.h"
#include "types.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

struct faidx_t;
struct bam1_t;
typedef struct htsFile samFile;

namespace kadayashi {

// For read downsampling of phasing & varcall pileups, and
// the pileup of kadayashi feature matrix generator.
constexpr int DOWNSAMPLE_WINDOW = 1000;  // store read depths every 1kb
constexpr int DOWNSAMPLE_READCAP = 200;  // target 200x

/**
 * @brief  Make a hashtable that maps read names to read haptags
 *         given the chunk.
 * @param ck   The chunk's full pileup info.
 */
std::unordered_map<std::string, int> kadayashi_local_haptagging_gen_ht(chunk_t &ck);

/**
 * @brief Perform phasing for a query region with the simple phasing method (flip-flop),
 *        return all info (read haptags, the chunk, and phasing breakpoints).
 *
 * This function is exposed for CLI use. To obtain only the read haptags, use
 * `kadayashi_simple_single_region_wrapper` instead.
 *
 * @remarks
 * Thread safety: Seeking the BAM file from other threads with
 * htslib *itr_query methods will produce unpredictable behavior.
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
    int cov_tot_all = 0;
    int cov_tot_phased = 0;

    /// forward stand coverage (regardless of phase)
    int cov_fwd = 0;
    /// reverse stand coverage (regardless of phase)
    int cov_bwd = 0;
};

/// A variant candidate allele.
struct vc_allele_t {
    /// read depth
    cov_t cov;

    /// the allele's sequence in u8i, where the last slot is cigar op.
    std::vector<uint8_t> allele = {};
};

constexpr uint8_t FLAG_VAR_NA = 0;
constexpr uint8_t FLAG_VAR_HOM = 1;
constexpr uint8_t FLAG_VAR_HET = 2;
constexpr uint8_t FLAG_VAR_MULTHET = 4;

constexpr uint8_t FLAG_VARSTAT_MAYBE = 0;  // temporary state
constexpr uint8_t FLAG_VARSTAT_ACCEPTED = 1;
constexpr uint8_t FLAG_VARSTAT_REJECTED = 2;
constexpr uint8_t FLAG_VARSTAT_UNSURE = 4;
constexpr uint8_t FLAG_VARSTAT_UNKNOWN = 8;

struct vc_variants1_val_t {
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
chunk_t variant_pileup_ht(dorado::secondary::BamFileView &hf,
                          const variants_t &ht_refvars,
                          const faidx_t *fai,
                          const str2int_t *qname2hp,
                          const std::string_view refname,
                          const uint32_t itvl_start,
                          const uint32_t itvl_end,
                          const pileup_pars_t &pp);

/**
 * @brief Phase reads in a query region. Produces a hashtable mapping
 *        from read names to their haptags in 0-index via the heuristics of
 *        iterative update of haptags of variants and reads.
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
        const float max_gapcompressed_seqdiv);

/// Exposed for CLI.
struct variant_fullinfo_t {
    bool is_valid{true};
    bool is_confident{true};
    bool is_multi_allele{false};
    int ref_depth{0};
    int min_depth{0};
    int median_depth{0};

    uint32_t pos0{0};  // 0-index
    uint8_t qual0{60};
    std::string ref_allele_seq0{};
    std::string alt_allele_seq0{};
    bool is_phased0{false};
    char genotype0[3] = {'0', '/', '0'};
    int alt_depth0{0};

    uint32_t pos1{0};  // 0-index
    uint8_t qual1{60};
    std::string ref_allele_seq1{};
    std::string alt_allele_seq1{};
    bool is_phased1{false};
    char genotype1[3] = {'0', '/', '0'};
    int alt_depth1{0};
};

std::string variant_fullinfo_to_string(const variant_fullinfo_t &v);

dorado::secondary::Variant convert_fullinfo_var_to_dorado_style(const variant_fullinfo_t &var,
                                                                bool keep_phasing);

bool operator==(const variant_fullinfo_t &a, const variant_fullinfo_t &b);

struct varcall_result_internal_t {
    std::unordered_map<std::string, int> qname2hp{};  // 0-index
    std::vector<variant_fullinfo_t> variants{};
    std::unordered_map<uint32_t, uint8_t> phasing_breakpoints{};
};

struct varcall_result_t {
    std::unordered_map<std::string, int> qname2hp{};     // 0-index
    std::vector<dorado::secondary::Variant> variants{};  // 0-index, for VCF
    std::unordered_map<uint32_t, uint8_t> phasing_breakpoints{};
    std::vector<std::pair<uint32_t, uint32_t>> read_endpoints{};
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
                                                    const std::string_view readgroup,
                                                    const bool disable_interval_expansion,
                                                    const int min_base_quality,
                                                    const int min_varcall_coverage,
                                                    const float min_varcall_fraction,
                                                    const int max_clipping,
                                                    const int min_strand_cov,
                                                    const float min_strand_cov_frac,
                                                    const float max_gapcompressed_seqdiv,
                                                    const bool ambig_ref);

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
 * @param ambig_ref  If set, include variants containing ambiguity bases.
 *                   Otherwise, no variants containing ambiguity codes in the
 *                   ref or alt alleles will be returned.
 */
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
                                                     const bool ambig_ref);

typedef dorado::secondary::ReadAlignmentData MedakaFeatureMatrix;
/*
 - Values: 0=uninitiated, 1234=ACGT, 5=del(or inside minor columns)
   If at position X, read#1 has an insertion of length five and read#2 has ins of
    length three, read#2's insertion will be like [55111] (right-aligned) or
    [11155] (left-aligned).
 - Major positions are aboslute coordinate wrt the reference sequence.
 - Minor positions are 0-valued except for in insertion.
 - Reference sequence is not in the matrix.
 - Reads in a lane are separated by at least five consecutive value 0 entries.
 - Example:
        pos 0123456789
        ref AACCCGGGGG
        r1  AA---GGGGG
                ^TT (insertion)
                  ^TT
        r2  AA--CGGG
                  ^TTT
    matrix will be:
      major 012344456666789
      minor 000001200123000
      ref   AACCC--GG---GGG
      lane1 115554433544300
      lane2 115525533444300
 */

enum class haplotagsource { FORCE_UNPHASED, USE_BAM_HAP_TAG, USE_TAG_FROM_HASHTABLE };

struct MedakaFeatureMatrixOptions {
    bool include_dwells;
    bool include_haplotype_column;
    bool include_snp_qv;
    int min_mapq;

    const int64_t num_dtypes;  // medaka: this should be 1 when `dtypes` is empty
    const std::vector<std::string>
            &dtypes;  // is used to retrieve data from DT tag, if num_types>1 then
                      // we will have extra feature dimensions in the mat.
    const std::string &tag_name;
    const int32_t tag_value;
    const bool tag_keep_missing;
    std::string readgroup;
    bool disable_read_packing;  // medaka: row_per_read ; note that enabling this
                                // will NOT automatically set `max_reads` to a large value.
                                // Even if we set `max_reads` to be big manually,
                                // the original feature mat gen has a hardcoded cap at 100 too
                                // which is included in this impl. Therefore if a query region
                                // has more than 100 reads and read packing is disabled,
                                // reads closer to the right side of the region will simply
                                // not be included.
    haplotagsource hap_source;
    int max_reads;
    bool right_align_insertions;
    double min_snp_accuracy;
};

/**
 * @brief Given access to aligned reads, a query interval and
 * optionally a set of desirable read haptags, return a read pileup matrix
 * of the scheme of `dorado::secondary::ReadAlignmentData`.
 * The returned major positions contain only reference positions covered by
 * reads retained in the output lanes. Consequently, `positions_major` may
 * contain discontinuities.
 *
 * @par Thread safety
 * Bam file.
 *
 * @param hf      Bam file access for aligned reads.
 * @param refname Query region's reference sequence name.
 * @param itvl_start Start of query region, 0-index inclusive.
 * @param itvl_end   End of query region,   0-index exclusive.
 * @param qname2hp   Mapping from read names to their haptags.
 * @param options    Options and hyperparameters for the read pileup matrix.
 *                   Note `hap_source` which can be used to ignore `qname2hp`.
 */
MedakaFeatureMatrix gen_medaka_feature_matrix(
        dorado::secondary::BamFileView &hf,
        std::string_view refname,
        const uint32_t itvl_start,
        const uint32_t itvl_end,
        const std::unordered_map<std::string, int32_t> &qname2hp,
        const MedakaFeatureMatrixOptions &options);

/**
 * @brief Wrapper of gen_medaka_feature_matrix (BamFile rather than BamFileView).
 * Given access to aligned reads, a query interval and
 * optionally a set of desirable read haptags, return a read pileup matrix
 * of the scheme of `dorado::secondary::ReadAlignmentData`.
 * The returned major positions contain only reference positions covered by
 * reads retained in the output lanes. Consequently, `positions_major` may
 * contain discontinuities.
 *
 * @par Thread safety
 * Bam file.
 *
 * @param hf      Bam file access for aligned reads.
 * @param refname Query region's reference sequence name.
 * @param itvl_start Start of query region, 0-index inclusive.
 * @param itvl_end   End of query region,   0-index exclusive.
 * @param qname2hp   Mapping from read names to their haptags.
 * @param options    Options and hyperparameters for the read pileup matrix.
 *                   Note `hap_source` which can be used to ignore `qname2hp`.
 */
MedakaFeatureMatrix gen_medaka_feature_matrix_wrapper(
        dorado::secondary::BamFile &bam_file,
        const std::string &refname,
        uint32_t itvl_start,
        uint32_t itvl_end,
        const std::unordered_map<std::string, int32_t> &qname2hp,
        const MedakaFeatureMatrixOptions &options);

void print_medaka_feature_matrix(const std::filesystem::path &out, const MedakaFeatureMatrix &mfm);
std::string print_medaka_feature_matrix(const MedakaFeatureMatrix &mfm);

}  // namespace kadayashi
