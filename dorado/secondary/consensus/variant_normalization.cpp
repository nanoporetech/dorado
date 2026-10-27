#include "secondary/consensus/variant_normalization.h"

#include "secondary/consensus/variant_calling.h"
#include "utils/log_utils.h"
#include "utils/sequence_utils.h"
#include "utils/stl_utils.h"

#include <algorithm>

namespace dorado::secondary {

std::pair<int64_t, int64_t> find_common_prefix_suffix_len(
        const std::string_view ref_allele,
        const std::vector<std::string>& alt_alleles,
        const bool non_empty) {
    if (std::empty(alt_alleles)) {
        return {0, 0};
    }

    const int64_t min_len = [&]() {
        int64_t ret = std::ssize(ref_allele);
        for (const auto& alt : alt_alleles) {
            ret = std::min<int64_t>(ret, std::ssize(alt));
        }
        return ret;
    }();

    const int64_t non_empty_offset = non_empty;

    // Find the maximum common suffix.
    int64_t suffix_trim = 0;
    {
        while (((suffix_trim + non_empty_offset) < min_len) &&
               std::all_of(std::begin(alt_alleles), std::end(alt_alleles),
                           [&](const std::string& alt) {
                               return (alt[std::ssize(alt) - 1 - suffix_trim] ==
                                       ref_allele[std::ssize(ref_allele) - 1 - suffix_trim]);
                           })) {
            ++suffix_trim;
        }
    }

    // Find the maximum common prefix after trimming the common suffix.
    int64_t prefix_trim = 0;
    {
        // Min len is shorter by the length of the common suffix.
        const int64_t new_min_len = min_len - suffix_trim;

        while (((prefix_trim + non_empty_offset) < new_min_len) &&
               std::all_of(std::begin(alt_alleles), std::end(alt_alleles),
                           [&](const std::string& alt) {
                               return (alt[prefix_trim] == ref_allele[prefix_trim]);
                           })) {
            ++prefix_trim;
        }
    }

    return std::make_pair(prefix_trim, suffix_trim);
}

std::pair<Variant, NormalizationStatus> normalize_variant_vs_ref_core(
        const Variant& variant,
        const std::string_view ref_seq,
        const int64_t win_start_offset,
        const bool allow_ambig_ref) {
    // The ref_seq starts at win_start_offset, while the variant coordinates are absolute.
    const int64_t win_end = win_start_offset + std::ssize(ref_seq);

    const auto trim_variant = [](Variant& var, const bool trim_start, const bool trim_end,
                                 const bool non_empty, const int64_t max_trim) -> bool {
        /**
         * \brief The trim_variant lambda returns true if any of the bases have been trimmed.
         * \param max_trim If < 0, all common prefix/suffix bases will be trimmed. Otherwise,
         *                  at most this many bases will be trimmed from both the prefix and suffix.
         */

        auto [prefix_len, suffix_len] = find_common_prefix_suffix_len(var.ref, var.alts, non_empty);

        if (!trim_start) {
            prefix_len = 0;
        }

        if (!trim_end) {
            suffix_len = 0;
        }

        if (max_trim >= 0) {
            prefix_len = std::min(prefix_len, max_trim);
            suffix_len = std::min(suffix_len, max_trim);
        }

        if ((prefix_len == 0) && (suffix_len == 0)) {
            return false;
        }

        var.ref = var.ref.substr(prefix_len, std::ssize(var.ref) - prefix_len - suffix_len);

        for (auto& seq : var.alts) {
            seq = seq.substr(prefix_len, std::ssize(seq) - prefix_len - suffix_len);
        }

        var.pos += prefix_len;

        return true;
    };

    const auto check_empty_alleles = [](const Variant& var) {
        return std::empty(var.ref) ||
               std::any_of(std::cbegin(var.alts), std::cend(var.alts),
                           [](const std::string_view s) { return std::empty(s); });
    };

    const auto prepend_base = [&](Variant& var, const bool allow_ambig) {
        const int64_t pos = var.pos - 1;
        if ((pos < win_start_offset) || (pos >= win_end)) {
            return false;
        }
        // const std::string base(ref_seq.substr(pos, 1));
        const char base = ref_seq[pos - win_start_offset];
        if (!allow_ambig && !utils::is_canonical_base(base)) {
            return false;
        }
        var.ref = base + var.ref;  // NOLINT(performance-inefficient-string-concatenation)
        for (std::string& alt : var.alts) {
            alt = base + alt;  // NOLINT(performance-inefficient-string-concatenation)
        }
        var.pos = pos;
        return true;
    };

    const auto append_base = [&](Variant& var, const bool allow_ambig) {
        const int64_t pos = var.pos + std::ssize(var.ref);
        if ((pos < win_start_offset) || (pos >= win_end)) {
            return false;
        }
        const char base = ref_seq[pos - win_start_offset];
        if (!allow_ambig && !utils::is_canonical_base(base)) {
            return false;
        }
        var.ref += base;
        for (std::string& alt : var.alts) {
            alt += base;
        }
        return true;
    };

    if (is_reference_record(variant)) {
        return {variant, NormalizationStatus::UNCHANGED};
    }

    if (win_start_offset < 0) {
        DORADO_LOG(warn, "Window start offset must not be negative. win_start_offset = {}",
                   win_start_offset);
        return {variant, NormalizationStatus::ERROR_NEGATIVE_OFFSET};
    }
    // REF can be empty for an unanchored insertion (inserted before variant.pos), which can then
    // also be located at the window end.
    const int64_t ref_end = variant.pos + std::ssize(variant.ref);
    if ((variant.pos < win_start_offset) || (variant.pos > win_end) || (ref_end > win_end)) {
        DORADO_LOG(warn,
                   "Variant is outside the reference sequence. variant.pos = {}, "
                   "ref_end = {}, win_start_offset = {}, ref_seq.size() = {}",
                   variant.pos, ref_end, win_start_offset, std::ssize(ref_seq));
        return {variant, NormalizationStatus::ERROR_OUT_OF_RANGE};
    }

    Variant ret = variant;
    NormalizationStatus ret_status;

    // Validate the REF sequence.
    const std::string_view actual_ref_subseq =
            ref_seq.substr(ret.pos - win_start_offset, std::size(ret.ref));
    if (ret.ref != actual_ref_subseq) {
        DORADO_LOG(warn,
                   "Variant REF does not match the reference sequence. ret.ref = \"{}\", "
                   "actual_ref_subseq = \"{}\"",
                   ret.ref, actual_ref_subseq);
        ret_status = NormalizationStatus::ERROR_REF;
        return {ret, ret_status};
    }

    // Validate that none of the ALTs match the reference.
    for (const std::string& alt : ret.alts) {
        if (alt == ret.ref) {
            DORADO_LOG(warn,
                       "Alternate alleles must differ from REF. ret.ref = "
                       "\"{}\", alt = \"{}\"",
                       ret.ref, alt);
            ret_status = NormalizationStatus::ERROR_DUPLICATE_ALLELE;
            return {ret, ret_status};
        }
    }

    // Whether the variant was modified in any of the steps below.
    bool modified = false;

    // Trim the end and push left.
    bool changed = false;
    do {
        Variant bak = ret;

        // Temporarily allow empty alleles so that indels can shift left.
        changed = trim_variant(ret, false, true, false, 1);

        // If any allele is empty, we need to move left if we can.
        if (check_empty_alleles(ret)) {
            const bool rv = prepend_base(ret, allow_ambig_ref);
            changed = changed || rv;
        }

        // If there are still empty alleles, revert the variant.
        if (check_empty_alleles(ret)) {
            ret = bak;
            changed = false;
            break;
        }

        modified = modified || changed;

    } while (changed);

    // Finally, if left extension did not help, extend right if possible.
    if (check_empty_alleles(ret)) {
        const bool is_appended = append_base(ret, allow_ambig_ref);
        modified = is_appended || modified;
    }

    // If both flanking bases are ambiguous, pad with an ambiguous base rather than leaving an empty
    // allele, which cannot be represented in the VCF.
    if (check_empty_alleles(ret) && !allow_ambig_ref) {
        const bool is_padded = prepend_base(ret, true) || append_base(ret, true);
        modified = is_padded || modified;
    }

    // Trim from the start.
    const bool is_trimmed = trim_variant(ret, true, false, true, -1);

    modified = is_trimmed || modified;

    ret_status = modified ? NormalizationStatus::CHANGED : NormalizationStatus::UNCHANGED;

    return {std::move(ret), ret_status};
}

std::pair<std::vector<Variant>, NormalizationStatus> normalize_variants_vs_ref_core(
        const std::span<const Variant> variants,
        const std::string_view ref_seq,
        const int64_t win_start_offset,
        const bool allow_ambig_ref) {
    if (std::empty(variants)) {
        return {};
    }

    std::vector<Variant> ret;
    ret.reserve(std::size(variants));
    bool modified = false;  // Whether any of the variants was modified.

    const int64_t seq_id = variants.front().seq_id;

    // Flag the input variants which overlap any other input variant. These are not normalized,
    // because normalizing them independently could change the alternate sequence they represent
    // together. A variant overlaps a later one only if it overlaps the next one (input is sorted).
    const std::vector<bool> is_overlapping = [&]() {
        std::vector<bool> ret_ovl(std::size(variants), false);
        int64_t furthest_end = win_start_offset;
        for (int64_t i = 0; i < std::ssize(variants); ++i) {
            const int64_t end = variant_end(variants[i]);
            const bool overlaps_prev = (variants[i].pos < furthest_end);
            const bool overlaps_next =
                    ((i + 1) < std::ssize(variants)) && (variants[i + 1].pos < end);
            ret_ovl[i] = overlaps_prev || overlaps_next;
            furthest_end = std::max(furthest_end, end);
        }
        return ret_ovl;
    }();

    const auto copy_input = [&variants](const NormalizationStatus var_status) {
        std::vector<Variant> input_vec{std::begin(variants), std::end(variants)};
        return std::pair{std::move(input_vec), var_status};
    };

    // Furthest end of all previous output (normalized) variants. Each variant can be shifted left
    // until it reaches the previous variants after they were normalized.
    int64_t prev_end = win_start_offset;

    for (int64_t i = 0; i < std::ssize(variants); ++i) {
        const Variant& v = variants[i];
        const int64_t prev_start = (i > 0) ? variants[i - 1].pos : win_start_offset;
        const int64_t next_start = ((i + 1) < std::ssize(variants))
                                           ? variants[i + 1].pos
                                           : (std::ssize(ref_seq) + win_start_offset);

        // Check that the variants are sorted.
        if ((v.seq_id != seq_id) || (v.pos < prev_start)) {
            DORADO_LOG(warn,
                       "Variants must be sorted on a single reference sequence for normalization.");
            return copy_input(NormalizationStatus::ERROR_NOT_SORTED);
        }

        if (is_overlapping[i]) {
            // Variant overlaps another input variant. Leave it as is.
            ret.emplace_back(v);
        } else if (is_reference_record(v)) {
            // Do not normalize reference records but keep them.
            ret.emplace_back(v);
        } else {
            // Normalize the variant within [prev_end, next_start). It does not overlap any other
            // input variant, so it lies entirely within this window.
            const std::string_view local_ref =
                    ref_seq.substr(prev_end - win_start_offset, next_start - prev_end);

            auto [var, var_status] =
                    normalize_variant_vs_ref_core(v, local_ref, prev_end, allow_ambig_ref);

            // If there was an error normalizing any of the variants, copy everything and
            // return unnormalized.
            if ((var_status != NormalizationStatus::CHANGED) &&
                (var_status != NormalizationStatus::UNCHANGED)) {
                return copy_input(var_status);
            }

            ret.emplace_back(std::move(var));

            modified = modified || (var_status == NormalizationStatus::CHANGED);
        }

        prev_end = std::max(prev_end, variant_end(ret.back()));
    }

    const NormalizationStatus ret_status =
            modified ? NormalizationStatus::CHANGED : NormalizationStatus::UNCHANGED;

    return std::pair{std::move(ret), ret_status};
}

std::vector<Variant> normalize_variants_vs_ref(const std::span<const Variant> variants,
                                               const std::string_view ref_seq,
                                               const bool allow_ambig_ref) {
    return normalize_variants_vs_ref_core(variants, ref_seq, 0, allow_ambig_ref).first;
}

std::vector<Variant> normalize_variants_vs_haplotypes(
        const std::string_view draft,
        const std::string_view ref_with_gaps,
        const std::vector<ConsensusResult>& cons_seqs_with_gaps,
        const std::vector<int64_t>& positions_major,
        const std::vector<int64_t>& positions_minor,
        const std::unordered_set<char>& symbol_set,
        const std::span<const Variant> variants,
        const bool allow_ambig_ref,
        const at::Tensor& probs_3D,
        const std::array<int32_t, 256>& symbol_lookup,
        const bool merge_overlapping,
        const bool merge_adjacent,
        const at::Tensor& basecalls,
        const float pass_min_qual,
        const float pass_min_gq,
        const float pass_min_aq,
        const float pass_min_vq) {
    constexpr int32_t MAX_NORMALIZATION_ITERATIONS = 10;

    if (std::empty(variants)) {
        return {};
    }

    if (std::empty(positions_major)) {
        return {};
    }

    if (std::size(positions_major) != std::size(positions_minor)) {
        throw std::runtime_error{
                "Number of elements of positions_major and positions_minor differs. "
                "positions_major.size() = " +
                std::to_string(std::size(positions_major)) +
                ", positions_minor.size() = " + std::to_string(std::size(positions_minor))};
    }

    // Start/end coordinate of the window.
    const int64_t win_start = positions_major.front();
    const int64_t win_end = positions_major.back() + 1;

    // Reference of the window. It is taken from the draft rather than from the reference columns of
    // the window, because a reference position can be present only through its insertion columns
    // (e.g. a window starting on an insertion column), and construct_variant anchors such variants
    // on the draft base.
    const std::string_view ref_seq = draft.substr(win_start, win_end - win_start);

    // Normalizes the variants within the window. The column span of each variant is extended to
    // cover both the original and the normalized variant. Returns {variants, modified}.
    const auto normalize_window = [&](const std::span<const Variant> in) {
        auto [out, out_status] =
                normalize_variants_vs_ref_core(in, ref_seq, win_start, allow_ambig_ref);
        for (Variant& var : out) {
            if (!is_reference_record(var)) {
                var.rstart = std::min(var.rstart,
                                      utils::find_column_to_insert(positions_major, var.pos));
                var.rend = std::max(
                        var.rend, utils::find_column_to_insert(positions_major, variant_end(var)));
            }
        }
        return std::make_pair(std::move(out), out_status == NormalizationStatus::CHANGED);
    };

    // Constructs a single variant from the haplotypes over the columns [rstart, rend). Empty alleles
    // (e.g. deletions) are padded by the normalization which follows.
    const auto construct_merged_variant = [&](const int64_t rstart, const int64_t rend,
                                              const Variant& first) {
        const std::string variant_source = utils::get_value_or_default(
                first.format, FORMAT_TAG_VARIANT_SOURCE, VARIANT_SOURCE_UNKNOWN);

        return construct_variant(draft, positions_major, positions_minor, ref_with_gaps,
                                 cons_seqs_with_gaps, first.seq_id, rstart, rend, true,
                                 allow_ambig_ref, variant_source, probs_3D, symbol_set,
                                 symbol_lookup, basecalls, pass_min_qual, pass_min_gq, pass_min_aq,
                                 pass_min_vq);
    };

    const auto has_disallowed_ambig_ref = [&](const int64_t start, const int64_t end) {
        return !allow_ambig_ref && (end > start) &&
               !is_subset_of_symbols(symbol_set, ref_with_gaps.substr(start, end - start));
    };

    // Replaces each group of overlapping and/or adjacent variants with a single variant.
    // Does not merge variants which cover ambiguous bases unless allow_ambig_ref is true.
    // Reference records are never merged. The group is extended greedily, one variant at a time.
    const auto merge_groups = [&](std::vector<Variant> in, const bool merge_overlapping_,
                                  const bool merge_adjacent_) {
        std::vector<Variant> out;
        out.reserve(std::size(in));
        for (int64_t i = 0; i < std::ssize(in);) {
            int64_t group_start = in[i].pos;
            int64_t group_end = variant_end(in[i]);
            int64_t group_rend = in[i].rend;

            bool prev_is_ambig = has_disallowed_ambig_ref(in[i].rstart, in[i].rend);

            int64_t j = i + 1;

            for (j = i + 1; j < std::ssize(in); ++j) {
                const bool curr_is_ambig = has_disallowed_ambig_ref(in[j].rstart, in[j].rend);

                const bool is_mergeable = !is_reference_record(in[i]) &&
                                          !is_reference_record(in[j]) &&
                                          ((merge_overlapping_ && (in[j].pos < group_end)) ||
                                           (merge_adjacent_ && (in[j].pos == group_end))) &&
                                          !prev_is_ambig && !curr_is_ambig;

                prev_is_ambig = curr_is_ambig;

                if (!is_mergeable) {
                    break;
                }

                group_start = std::min(group_start, in[j].pos);
                group_end = std::max(group_end, variant_end(in[j]));
                group_rend = std::max(group_rend, in[j].rend);
            }

            if (j == (i + 1)) {
                // No merging happened, just move it.
                out.emplace_back(std::move(in[i]));

            } else {
                const int64_t new_rstart =
                        utils::find_column_to_insert(positions_major, group_start);
                const int64_t new_rend = std::max(
                        group_rend, utils::find_column_to_insert(positions_major, group_end));

                // Variants begin on reference positions, so the start column is the column of the
                // group start. The end column covers the furthest reach of all variants in the group.
                Variant var = construct_merged_variant(new_rstart, new_rend, in[i]);

                // Empty alleles are kept, because they are padded by the normalization which follows.
                // If merging failed (e.g. an ambiguous ref base with ambig_ref = false), keep the
                // original variants of the group.
                if (!is_reference_record(var)) {
                    // Add the merged variant.
                    out.emplace_back(std::move(var));

                } else {
                    // Failed for some reason, keep the originals.
                    for (int64_t k = i; k < j; ++k) {
                        out.emplace_back(std::move(in[k]));
                    }
                }
            }

            i = j;
        }
        return out;
    };

    // Construct the variants.
    std::vector<Variant> ret = normalize_window(variants).first;

    // Merge and normalize again until convergence, only if required.
    if (merge_overlapping || merge_adjacent) {
        int32_t iter = 0;
        for (iter = 0; iter < MAX_NORMALIZATION_ITERATIONS; ++iter) {
            // Merging replaces each group of two or more variants with at most one variant, so
            // variants were merged only if the count decreased.
            std::vector<Variant> merged = merge_groups(ret, merge_overlapping, merge_adjacent);
            const bool any_merged = (std::size(merged) != std::size(ret));
            auto [normalized, any_normalized] = normalize_window(merged);
            ret = std::move(normalized);
            if (!any_merged && !any_normalized) {
                break;
            }
        }
        if (iter == MAX_NORMALIZATION_ITERATIONS) {
            const int32_t seq_id = (!std::empty(variants)) ? (variants.front().seq_id) : -1;
            DORADO_LOG(debug,
                       "Reached MAX_NORMALIZATION_ITERATIONS in normalize_variants_vs_haplotypes. "
                       "seq_id = {}, win_start = {}, win_end = {}",
                       seq_id, win_start, win_end);
        }
    }

    return ret;
}

}  // namespace dorado::secondary
