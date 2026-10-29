#pragma once

#include "secondary/common/variant.h"
#include "secondary/consensus/consensus_result.h"

#include <ATen/ATen.h>

#include <array>
#include <cstdint>
#include <span>
#include <string_view>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace dorado::secondary {

/**
 * \brief Result of a normalization. CHANGED and UNCHANGED indicate success; ERROR_* values indicate
 *          invalid input, in which case the input is returned unchanged.
 */
enum class NormalizationStatus {
    CHANGED,
    UNCHANGED,
    ERROR_REF,
    ERROR_DUPLICATE_ALLELE,
    ERROR_NOT_SORTED,
    ERROR_NEGATIVE_OFFSET,
    ERROR_OUT_OF_RANGE,
};

/**
 * \brief Finds the longest common prefix and suffix for a given reference sequence and a set of alleles.
 * \param ref_allele Sequence of the reference allele for this variant.
 * \param alt_alleles A vector of sequences for all alternate alleles.
 * \param non_empty If true, alleles will never be trimmed such that they end up empty (at least one base will remain).
 *                  If false, one or more alleles can fully be trimmed.
 * \returns {prefix length, suffix length}, with the suffix taking precedence. No ALTs means no trimming.
 */
std::pair<int64_t, int64_t> find_common_prefix_suffix_len(
        std::string_view ref_allele,
        const std::vector<std::string>& alt_alleles,
        bool non_empty);

/**
 * \brief Normalize a sequence variant against a gapless reference subsequence. Normalization never
 *          extends outside of ref_seq, so ref_seq should cover only the bases which the variant can use
 *          (e.g. the bases between its neighboring variants).
 *        Reference records are returned unchanged without validation or normalization.
 * \param variant Variant to normalize. A copy of this object is produced and returned (normalized or not).
 * \param ref_seq Reference subsequence within which the variant is normalized. ref_seq[0] is at the
 *          chromosome position win_start_offset.
 * \param win_start_offset Chromosome position of the first base of ref_seq. Must not be negative.
 * \param allow_ambig_ref If true, allow normalization over ambiguous reference bases.
 * \returns {variant, status}. The status is CHANGED or UNCHANGED on success. On invalid input, the input
 *          variant is returned with ERROR_REF if REF mismatches ref_seq (including an empty REF), or with
 *          ERROR_DUPLICATE_ALLELE if an ALT equals REF, with ERROR_OUT_OF_RANGE if the REF span lies
 *          outside of ref_seq, or with ERROR_NEGATIVE_OFFSET if win_start_offset is negative.
 *          Reference records are returned as UNCHANGED without validation.
 * \details Variant coordinates are absolute (chromosome) coordinates. Empty alleles are padded on the
 *          left when possible, otherwise on the right. Non-reference records must have ALTs distinct
 *          from REF, including after case folding.
 */
std::pair<Variant, NormalizationStatus> normalize_variant_vs_ref_core(const Variant& variant,
                                                                      std::string_view ref_seq,
                                                                      int64_t win_start_offset,
                                                                      bool allow_ambig_ref);

/**
 * \brief Same as normalize_variants_vs_ref, but ref_seq covers only a window of the chromosome which
 *          starts at win_start_offset. Variant coordinates remain absolute (chromosome) coordinates,
 *          and normalization never extends outside of the window. Each variant is shifted left at most
 *          until the previous variants after their normalization, and padded on the right at most until
 *          the next input variant.
 * \param variants Input records sorted by position on a single chromosome. Reference records are
 *                 preserved and their entire spans act as barriers to left alignment. Records which
 *                 overlap any other input record are preserved unchanged.
 * \param ref_seq Reference sequence of the window. ref_seq[0] is at chromosome position win_start_offset.
 * \param win_start_offset Chromosome position of the first base of ref_seq. Must not be negative.
 * \param allow_ambig_ref If true, allow normalization over ambiguous reference bases.
 * \returns {variants, status}. The status is CHANGED if any variant was modified, UNCHANGED otherwise.
 *          On invalid input, all input variants are returned unchanged with ERROR_NOT_SORTED if records
 *          are unsorted, on different chromosomes, or start before win_start_offset, or with the error
 *          status of normalize_variant_vs_ref_core if any variant could not be normalized (e.g.
 *          ERROR_OUT_OF_RANGE if a variant lies outside of the window).
 **/
std::pair<std::vector<Variant>, NormalizationStatus> normalize_variants_vs_ref_core(
        std::span<const Variant> variants,
        std::string_view ref_seq,
        int64_t win_start_offset,
        bool allow_ambig_ref);

/**
 * \brief Takes a vector of variants and normalizes them with respect to a single reference sequence,
 *          while taking care not to produce overlapping variants.
 * \param variants Input records sorted by position on a single chromosome. Reference records are
 *                 preserved and their entire spans act as barriers to left alignment. Records which
 *                 overlap any other input record are preserved unchanged.
 * \param ref_seq The entire reference chromosome sequence, covering all possible positions of the input variants.
 * \param allow_ambig_ref If true, allow normalization over ambiguous reference bases.
 * \returns A vector of normalized variants. On invalid input (see normalize_variants_vs_ref_core), the
 *          input variants are returned unchanged and the error status is not reported.
 **/
std::vector<Variant> normalize_variants_vs_ref(std::span<const Variant> variants,
                                               std::string_view ref_seq,
                                               bool allow_ambig_ref);

/**
 * \brief Normalizes the variants of a single inference sample (window) against the reference, and merges
 *          the adjacent or overlapping ones into larger variants constructed from the haplotypes.
 *          Steps: (1) each variant is normalized without overlapping its neighbours, (2) groups of
 *          adjacent/overlapping variants are replaced by a single variant constructed with
 *          construct_variant over the union of their spans, (3) all variants are normalized again
 *          until convergence. Normalization never extends beyond the window.
 * \param draft The entire input draft/reference sequence.
 * \param ref_with_gaps Reference sequence with gaps at minor positions, one character per column.
 * \param cons_seqs_with_gaps Consensus sequence with gaps for each haplotype, one character per column.
 * \param positions_major Major (reference) coordinate for each column.
 * \param positions_minor Minor coordinate for each column (0 for reference positions).
 * \param symbol_set Set of valid symbols in the label scheme.
 * \param variants Input variants of this window, sorted by position.
 * \param allow_ambig_ref Allow ambiguous reference bases.
 * \param probs_3D Probabilities of shape [num_columns x num_haplotypes x num_classes].
 * \param symbol_lookup Class index for each symbol, or -1.
 * \param basecalls Optional basecalls tensor used to compute the AD tag. Can be undefined.
 * \returns Normalized, non-overlapping variants with updated rstart/rend columns.
 */
std::vector<Variant> normalize_variants_vs_haplotypes(
        std::string_view draft,
        std::string_view ref_with_gaps,
        const std::vector<ConsensusResult>& cons_seqs_with_gaps,
        const std::vector<int64_t>& positions_major,
        const std::vector<int64_t>& positions_minor,
        const std::unordered_set<char>& symbol_set,
        std::span<const Variant> variants,
        bool allow_ambig_ref,
        const at::Tensor& probs_3D,
        const std::array<int32_t, 256>& symbol_lookup,
        bool merge_overlapping,
        bool merge_adjacent,
        const at::Tensor& basecalls,
        float pass_min_qual,
        float pass_min_gq,
        float pass_min_aq,
        float pass_min_vq);

/**
 * \brief Normalization fallback for non-valid variants (e.g. with empty ALTs) which could not be
 *          normalized within their inference window. Each such variant is re-normalized against the
 *          reference within the bases between its neighboring variants. Variants which still have
 *          empty ALTs are converted to reference records (see sanitize_empty_alts).
 *        Non-valid variants are not removed here (see remove_non_valid_variants).
 * \param variants Variants sorted by position on a single chromosome. Modified in place.
 * \param ref_seq The entire reference chromosome sequence.
 * \param allow_ambig_ref If true, allow normalization over ambiguous reference bases.
 * \param ploidy Ploidy used for the GT of variants converted to reference records.
 */
void renormalize_non_valid_variants(std::vector<Variant>& variants,
                                    std::string_view ref_seq,
                                    bool allow_ambig_ref,
                                    int32_t ploidy);

}  // namespace dorado::secondary
