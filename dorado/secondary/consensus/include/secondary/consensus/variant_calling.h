#pragma once

#include "hts_utils/FastxRandomReader.h"
#include "sample.h"
#include "secondary/common/interval.h"
#include "secondary/common/region.h"
#include "secondary/common/variant.h"
#include "secondary/consensus/consensus_result.h"
#include "secondary/features/decoder_base.h"
#include "variant_calling_sample.h"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace dorado::secondary {

constexpr bool DEBUG_VC_SAMPLES = false;
constexpr int64_t DEBUG_VC_SAMPLES_POINT = -1;
constexpr int64_t DEBUG_VC_SAMPLES_START = DEBUG_VC_SAMPLES_POINT - 1000;
constexpr int64_t DEBUG_VC_SAMPLES_END = DEBUG_VC_SAMPLES_POINT + 1000;

constexpr float VCF_MAX_GQ_CAP = 60.0f;
constexpr float VCF_MAX_PL_CAP = 990.0f;
constexpr int32_t VAF_DP = 2;
constexpr std::array<std::pair<int64_t, float>, 2> DEFAULT_GVCF_REFERENCE_BLOCK_GQ_MARGINS{
        {{0, 0.5f}, {10, 5.0f}}};

/**
 * \brief Convert a float quality into an integer string.
 * \param qual Float QUAL value.
 */
std::string q_string_from_float(float qual);

/**
 * \brief Extend variant span in preparation for allele depth calculation.
 *        Updates the rstart, rend and alleles in place.
 *        Basic rules are:
 *        - include all minor positions immediately flanking the span
 *        - if the ends of a span are an indel, include the flanking major base
 * \param rstart Span start coordinate in positions space.
 * \param rend Span end coordinate in positions space (end-exclusive).
 * \param alleles Gap-compressed sequences.
 * \param prepended Whether a reference base has already been prepended; if true, no further changes will be made to rstart.
 * \param draft Reference sequence for the full contig.
 * \param positions_major Major reference coordinates.
 * \param positions_minor Minor coordinates.
 * \param ref_seq_with_gaps Gap-expanded reference sequence.
 * \param cons_seqs_with_gaps Gap-expanded consensus sequence(s).
 */
void extend_span(int64_t& rstart,
                 int64_t& rend,
                 std::vector<std::string>& alleles,
                 bool prepended,
                 const std::string_view draft,
                 const std::vector<int64_t>& positions_major,
                 const std::vector<int64_t>& positions_minor,
                 const std::string_view ref_seq_with_gaps,
                 const std::vector<ConsensusResult>& cons_seqs_with_gaps);

/**
 * \brief Construct a gVCF reference block.
 *        Handles setting end points and populating Variant::genotype fields.
 *        Also sets GT and GQ values for non-canonical blocks.
 * \param seq_id Reference sequence id.
 * \param pos Reference position.
 * \param ref_base Reference base at pos.
 * \param qual Quality score.
 * \param ploidy Number of haplotypes.
 * \param min_depth Minimum read depth in the block; set to -1 to represent missing data.
 * \param end Reference end position.
 * \param rstart Span start in inference chunk coordinates.
 * \param rend Span end in inference chunk coordinates.
 * \param variant_source Value for the SRC FORMAT tag (e.g. inference vs. plain reference block).
 */
Variant make_gvcf_reference_record(int32_t seq_id,
                                   int64_t pos,
                                   char ref_base,
                                   float qual,
                                   int32_t ploidy,
                                   int32_t min_depth,
                                   int64_t end,
                                   int64_t rstart,
                                   int64_t rend,
                                   std::string_view variant_source);

/**
 * \brief Add VAF field to a variant.
 *        The variant allele fraction is calculated from the alt AD values and depth
 *        in var.format, provided both are present, and the VAF field is added. If
 *        either AD or depth is missing, or the variant is a reference record with no
 *        alts, no VAF record is added.
 * \param var Variant, modified in place.
 */
void add_VAF(Variant& var);

/**
 * \brief Post-process variants with empty alts.
 *        Sets the variant to a reference block with no filter.
 * \param var Variant, modified in place.
 * \param ploidy Number of haplotypes.
 */
void sanitize_empty_alts(Variant& var, int32_t ploidy);

/**
 * \brief Reduce a polyploid variant to haploid.
 *          If require_hom is true, then the variant will be called if all alleles have the variant.
 *          Otherwise, the variant is called if any allele has the variant.
 *          If the variant is not called, it's alt is cleared so that it will be discarded when
 *          filtering to valid variants.
 *          If the number of unique alts is >1, the variant is discarded.
 * \param var Input polyploid variant.
 * \param require_hom Whether to require a homozygous call to retain the variant.
 * \returns New haploid variant.
 */
Variant collapse_to_haploid(const Variant& var, const bool require_hom);

/**
 * \brief Utility function which checks if every character of query is in the allowed
 *          symbol map.
 */
bool is_subset_of_symbols(const std::unordered_set<char>& symbol_map, const std::string_view query);

/**
 * \brief Constructs a variant from the haplotype sequences over a span of sample columns, and computes
 *          its genotype and quality scores from the per-haplotype probabilities.
 * \param draft The entire input draft/reference sequence.
 * \param positions_major Major (reference) coordinate for each sample column.
 * \param positions_minor Minor coordinate for each sample column (0 for reference positions).
 * \param ref_seq_with_gaps Reference sequence with gaps at minor positions, one character per column.
 * \param cons_seqs_with_gaps Consensus sequence with gaps for each haplotype, one character per column.
 * \param seq_id Reference sequence ID of the variant.
 * \param rstart Start column of the variant span. Zero-based.
 * \param rend End column of the variant span. Non-inclusive.
 * \param is_var If true, an empty variant is returned when no haplotype differs from the reference.
 * \param ambig_ref Allow ambiguous reference bases in the variant.
 * \param variant_source Value for the SRC FORMAT tag.
 * \param probs_3D Probabilities of shape [num_columns x num_haplotypes x num_classes].
 * \param symbol_set Set of valid symbols in the label scheme.
 * \param symbol_lookup Class index for each symbol, or -1.
 * \param basecalls Optional basecalls tensor used to compute the AD tag. Can be undefined.
 * \returns The constructed variant, or an empty (invalid) variant if no variant can be constructed.
 */
Variant construct_variant(std::string_view draft,
                          const std::vector<int64_t>& positions_major,
                          const std::vector<int64_t>& positions_minor,
                          std::string_view ref_seq_with_gaps,
                          const std::vector<ConsensusResult>& cons_seqs_with_gaps,
                          int32_t seq_id,
                          int64_t rstart,
                          int64_t rend,
                          bool is_var,
                          bool ambig_ref,
                          const std::string& variant_source,
                          const at::Tensor& probs_3D,
                          const std::unordered_set<char>& symbol_set,
                          const std::array<int32_t, 256>& symbol_lookup,
                          const at::Tensor& basecalls,
                          float pass_min_qual,
                          float pass_min_gq,
                          float pass_min_aq,
                          float pass_min_vq);

/**
 * \brief Decodes polyploid variants from a given tensor of probabilities for a single inference sample (not a batch).
 * \param decoder The decoder which contains the label scheme.
 * \param vc_sample A VariantCallingSample containing ref ID, positions and output scores.
 * \param draft The entire input draft/reference sequence.
 * \param ambig_ref Allow ambiguous reference bases (`N`) for variant calling.
 * \param return_all Returns gVCF records for all reference positions, including the non-variant ones.
 * \param merge_overlapping Overlapping variants will be merged and deduplicated.
 * \param merge_adjacent Variants which are immediately adjacent will be merged into a single larger variant.
 * \returns Normalized variants for the input sample. All variants of the sample are normalized together
 *          with normalize_variants_vs_haplotypes, after they are constructed from the haplotypes.
 */
std::vector<Variant> general_decode_variants(
        const DecoderBase& decoder,
        const VariantCallingSample& vc_sample,
        const std::string_view draft,
        const float pass_min_qual,
        const float pass_min_gq,
        const float pass_min_aq,
        const float pass_min_vq,
        bool ambig_ref,
        bool return_all,
        bool merge_overlapping,
        bool merge_adjacent,
        std::span<const std::pair<int64_t, float>> gvcf_reference_block_gq_margins);

/**
 * \brief Get the set of genotype combinations for (ploidy, num_alleles) from lookup table
 *        of pre-defined genotypes. Throws for invalid inputs.
 * \param ploidy Number of haplotypes.
 * \param num_alleles Number of alleles.
 * \returns Pair containing: (number of genotype combinations, genotype combinations).
 */
std::pair<int64_t, std::span<const std::vector<int32_t>>> lookup_gts(int64_t ploidy,
                                                                     int64_t num_alleles);

/**
 * \brief Utility function to split GT string back into allele indices.
 *        Handles `|` or `/` separators but not the mixed case described in the VCF spec.
 *        If both `|` and `/` are present, this function will throw.
 * \param gt Genotype string
 * \returns Pair containing: (Vector of allele strings in the order they appear in gt, whether the genotype is phased).
 */
std::pair<std::vector<std::string_view>, bool> split_GT_string(const std::string_view gt);

}  // namespace dorado::secondary
