#pragma once

#include "hts_utils/FastxRandomReader.h"
#include "sample.h"
#include "secondary/common/interval.h"
#include "secondary/common/region.h"
#include "secondary/common/variant.h"
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

constexpr float VCF_MAX_GQ_CAP = 60.0f;
constexpr std::array<std::pair<int64_t, float>, 2> DEFAULT_GVCF_REFERENCE_BLOCK_GQ_MARGINS{
        {{0, 0.5f}, {10, 5.0f}}};

/**
 * \brief Utility function to normalize the genotype information of a variant.
 *          For a given variant, it deduplicates and enumerates alt alleles, creates the
 *          GT and GQ tags and sets the filter tag.
 * \param var Input variant for normalization.
 * \param ploidy The ploidy of the dataset, needed to sanity check the number of alts.
 * \param min_qual Minimum variant quality to mark the variant as PASS.
 * \returns New variant with normalized genotyping information.
 */
Variant normalize_genotype(const Variant& var, const int32_t ploidy, const float min_qual);

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
 * \brief Normalizes the input variant (pushes it to the left). Stops normalization when another
 *          variant is reached, or until it can no longer be normalized.
 * \param ref_with_gaps Reference (or draft) sequence for the region specified by positions_major.
 *                      It has '*' added to the minor positions.
 * \param cons_seqs_with_gaps Consensus sequences, one for each haplotype. Minor positions can have
 *                              an actual predicted base or a '*' in them.
 * \param positions_major Major coordinates for each position in the input probs tensor (i.e. actual reference positions).
 * \param positions_minor Minor coordinates for each position in the input probs tensor (i.e. 0 if this is a major position,
 *                          1 and above for large insertion regions).
 * \param variant Input variant to be normalized.
 * \returns A normalized variant.
 **/
Variant normalize_variant(const std::string_view ref_with_gaps,
                          const std::vector<std::string_view>& cons_seqs_with_gaps,
                          const std::vector<int64_t>& positions_major,
                          const std::vector<int64_t>& positions_minor,
                          const std::unordered_set<char>& symbol_set,
                          const Variant& variant,
                          bool ambig_variant);

/**
 * \brief Decodes polyploid variants from a given tensor of probabilities for a single inference sample (not a batch).
 * \param decoder The decoder which contains the label scheme.
 * \param vc_sample A VariantCallingSample containing ref ID, positions and output scores.
 * \param draft The entire input draft/reference sequence.
 * \param ambig_ref Allow ambiguous reference bases (`N`) for variant calling.
 * \param return_all Returns gVCF records for all reference positions, including the non-variant ones.
 * \param normalize Normalizes the variants (pushes them to the left if possible).
 * \param merge_overlapping Overlapping variants will be merged and deduplicated.
 * \param merge_adjacent Variants which are immediately adjacent will be merged into a single larger variant.
 * \returns Normalized variants for the input sample.
 */
std::vector<Variant> general_decode_variants(
        const DecoderBase& decoder,
        const VariantCallingSample& vc_sample,
        const std::string_view draft,
        const float pass_min_qual,
        bool ambig_ref,
        bool return_all,
        bool normalize,
        bool merge_overlapping,
        bool merge_adjacent,
        std::span<const std::pair<int64_t, float>> gvcf_reference_block_gq_margins);

}  // namespace dorado::secondary
