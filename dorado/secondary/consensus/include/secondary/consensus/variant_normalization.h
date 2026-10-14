#pragma once

#include "secondary/common/variant.h"

#include <span>
#include <string_view>
#include <tuple>
#include <vector>

namespace dorado::secondary {

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
 * \returns {normalized variant, true if the variant was modified}.
 * \throws std::out_of_range If the input REF span lies outside of ref_seq.
 * \throws std::invalid_argument If win_start_offset is negative, if REF mismatches ref_seq, or if an ALT
 *          equals REF.
 * \details Variant coordinates are absolute (chromosome) coordinates. Empty alleles are padded on the
 *          left when possible, otherwise on the right. Non-reference records must have ALTs distinct
 *          from REF, including after case folding.
 */
std::pair<Variant, bool> normalize_variant_vs_ref_core(const Variant& variant,
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
 * \returns {normalized variants, true if any of the variants was modified}.
 * \throws std::invalid_argument If records are unsorted or on different chromosomes. Input
 *                              validation errors of normalize_variant_vs_ref_core also propagate.
 **/
std::pair<std::vector<Variant>, bool> normalize_variants_vs_ref_core(
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
 * \returns A vector of normalized variants.
 * \throws std::invalid_argument If records are unsorted or on different chromosomes. Input
 *                              validation errors also propagate.
 **/
std::vector<Variant> normalize_variants_vs_ref(std::span<const Variant> variants,
                                               std::string_view ref_seq,
                                               bool allow_ambig_ref);

}  // namespace dorado::secondary
