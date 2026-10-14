#include "secondary/consensus/variant_normalization.h"

#include "utils/sequence_utils.h"

#include <spdlog/fmt/bundled/format.h>

#include <algorithm>
#include <stdexcept>

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

std::pair<Variant, bool> normalize_variant_vs_ref_core(const Variant& variant,
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
        return {variant, false};
    }

    if (win_start_offset < 0) {
        throw std::invalid_argument(
                fmt::format("Window start offset must not be negative. win_start_offset = {}",
                            win_start_offset));
    }
    if ((variant.pos < win_start_offset) || (variant.pos >= win_end) ||
        (variant_end(variant) > win_end)) {
        throw std::out_of_range(fmt::format(
                "Variant is outside the reference sequence. variant.pos = {}, variant_end = {}, "
                "win_start_offset = {}, ref_seq.size() = {}",
                variant.pos, variant_end(variant), win_start_offset, std::ssize(ref_seq)));
    }

    Variant ret = variant;

    // Validate the REF sequence.
    const std::string_view actual_ref_subseq =
            ref_seq.substr(ret.pos - win_start_offset, variant_ref_span(ret));
    if (ret.ref != actual_ref_subseq) {
        throw std::invalid_argument(
                fmt::format("Variant REF does not match the reference sequence. ret.ref = \"{}\", "
                            "actual_ref_subseq = \"{}\"",
                            ret.ref, actual_ref_subseq));
    }

    // Validate that none of the ALTs match the reference.
    for (const std::string& alt : ret.alts) {
        if (alt == ret.ref) {
            throw std::invalid_argument(fmt::format(
                    "Alternate alleles must differ from REF. ret.ref = \"{}\", alt = \"{}\"",
                    ret.ref, alt));
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

    return {std::move(ret), modified};
}

std::pair<std::vector<Variant>, bool> normalize_variants_vs_ref_core(
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
            throw std::invalid_argument{
                    "Variants must be sorted on a single reference sequence for normalization."};
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
            auto [var, var_modified] =
                    normalize_variant_vs_ref_core(v, local_ref, prev_end, allow_ambig_ref);
            ret.emplace_back(std::move(var));
            modified = modified || var_modified;
        }

        prev_end = std::max(prev_end, variant_end(ret.back()));
    }

    return {std::move(ret), modified};
}

std::vector<Variant> normalize_variants_vs_ref(const std::span<const Variant> variants,
                                               const std::string_view ref_seq,
                                               const bool allow_ambig_ref) {
    return normalize_variants_vs_ref_core(variants, ref_seq, 0, allow_ambig_ref).first;
}

}  // namespace dorado::secondary
