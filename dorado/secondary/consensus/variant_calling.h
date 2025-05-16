#pragma once

#include "hts_io/FastxRandomReader.h"
#include "sample.h"
#include "secondary/features/decoder_base.h"
#include "secondary/interval.h"
#include "secondary/variant.h"
#include "variant_calling_sample.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace dorado::secondary {

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
                          const Variant& variant);

/**
 * \brief Decodes polyploid variants from a given tensor of probabilities for a single inference sample (not a batch).
 * \param decoder The decoder which contains the label scheme.
 * \param seq_id ID of the reference sequence from where the sample comes from. Needed to construct the Variant object.
 * \param positions_major Major coordinates for each position in the input probs tensor (i.e. actual reference positions).
 * \param positions_minor Minor coordinates for each position in the input probs tensor (i.e. 0 if this is a major position,
 *                          1 and above for large insertion regions).
 * \param probs Class probabilies, either haploid or polyploid, for a single sample (not batch). Legacy haploid shape:
 *              [num_positions x num_classes]. Current polyploid shape: [num_positions x num_haplotypes x num_classes].
 *              Number of classes corresponds to the number of symbols in the label scheme.
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
        int32_t seq_id,
        const std::vector<int64_t>& positions_major,
        const std::vector<int64_t>& positions_minor,
        const at::Tensor& probs,  // Probabilities for a single sample (not batch).
        const std::string_view draft,
        bool ambig_ref,
        bool return_all,
        bool normalize,
        bool merge_overlapping,
        bool merge_adjacent);

}  // namespace dorado::secondary
