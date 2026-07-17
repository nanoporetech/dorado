#include "secondary/consensus/variant_calling.h"

#include "secondary/consensus/consensus_result.h"
#include "secondary/consensus/consensus_utils.h"
#include "torch_utils/tensor_utils.h"
#include "utils/rle.h"
#include "utils/sequence_utils.h"

#include <ATen/ATen.h>
#include <cxxpool.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <span>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

// #define DEBUG_VC_DATA_DUMP
// #define DEBUG_VARIANT_REGIONS
// #define DEBUG_NORMALIZE_VARIANT

namespace dorado::secondary {

namespace {

std::string remove_gaps(const std::string_view seq) {
    std::string ret;
    ret.reserve(std::size(seq));
    std::copy_if(std::cbegin(seq), std::cend(seq), std::back_inserter(ret),
                 [](const char c) { return c != '*'; });
    return ret;
}

float round_float(float val, const int32_t decimal_places) {
    const float f = std::pow(10.0f, static_cast<float>(decimal_places));
    val = std::round(val * f) / f;
    return val;
}

bool is_subset_of_symbols(const std::unordered_set<char>& symbol_map,
                          const std::string_view query) {
    for (const char c : query) {
        if (symbol_map.count(c) == 0) {
            return false;
        }
    }
    return true;
}

std::string make_reference_gt(const int32_t ploidy, const char allele) {
    std::ostringstream oss_gt;
    for (int32_t i = 0; i < ploidy; ++i) {
        if (i > 0) {
            oss_gt << '/';
        }
        oss_gt << allele;
    }
    return oss_gt.str();
}

/**
 * \brief Compute the phred quality from a given probability, with a max QV cap.
 */
float phred(float err, const float cap) {
    err = std::clamp(err, std::pow(10.0f, -cap / 10.0f), 1.0f);
    const float q = -10.0f * std::log10(err);
    return std::min(q, cap);
}

/**
 * \brief Create a lookup table of valid symbols.
 */
std::array<int32_t, 256> create_symbol_lookup(const std::string_view symbols) {
    std::array<int32_t, 256> ret;
    std::fill(std::begin(ret), std::end(ret), -1);
    for (int32_t i = 0; i < std::ssize(symbols); ++i) {
        ret[static_cast<int32_t>(symbols[i])] = i;
    }
    return ret;
}

/**
 * \brief Utility function to compute a log probability of a subsequence occurring for
 *          a specified haplotype.
 * \param probs_3D A 3D tensor of probabilities (output of inference) for a single sample (not batch).
 *                  Dimensions: [seq_len x num_haplotypes x num_classes].
 * \param seq Input sequence, used to access class probabilities.
 * \param symbol_lookup Lookup table of symbol chars (bases) -> numeric ID of that symbol, to encode sequence bases into class IDs.
 * \param rstart Region start (start of the subsequence in seq). Zero-based.
 * \param rend Region end (end of the subsequence in seq). Non-inclusive.
 * \param hap_id Haplotype ID, needed to access class probabilities for that particular haplotype in the probs_3D tensor.
 * \param substitute_n If true, the `N` bases in the input sequence will be replaced with `*` symbols. Useful for computing the probability of a reference.
 * \returns A single value with the log probability of that subsequence occurring for the specified haplotype, given the class probabilities obtained through inference.
 */
float compute_subseq_log_prob(
        const at::Tensor& probs_3D,  // Tensor sliced to the [rstart, rend] range
        // const dorado::Span<const float> prob_data,
        const std::string_view seq,
        const std::array<int32_t, 256>& symbol_lookup,
        const int64_t rstart,
        const int64_t rend,
        const int64_t hap_id,
        const bool substitute_n) {
    if (std::size(probs_3D.sizes()) != 3) {
        throw std::runtime_error(
                "Tensor of probabilities given to compute_subseq_quality is of wrong shape. Input "
                "shape: " +
                utils::tensor_shape_as_string(probs_3D) + ", but expected 3 dimensions.");
    }

    const int64_t num_pos = probs_3D.size(0);
    const int64_t num_haplotypes = probs_3D.size(1);
    const int64_t num_classes = probs_3D.size(2);

    const std::span<const float> data(probs_3D.data_ptr<float>(),
                                      static_cast<size_t>(num_pos * num_haplotypes * num_classes));

    if (std::empty(data)) {
        return 0.0f;
    }

    float total = 0.0f;
    for (int64_t pos = rstart; pos < rend; ++pos) {
        const char base = (substitute_n && (seq[pos] == 'N')) ? '*' : seq[pos];
        const int32_t class_id = symbol_lookup[base];

        const int64_t idx = pos * num_haplotypes * num_classes + hap_id * num_classes + class_id;
        const float prob = std::max(1e-10f, data[idx]);
        const float log_prob = std::log(prob);

        total += log_prob;
    }

    return total;
}

/**
 * \brief Utility function to compute the phred-scaled log-likelihood of a given set of sequences
 *          occurring for any input haplotype.
 * \param probs_3D A 3D tensor of probabilities (output of inference) for a single sample (not batch).
 *                  Dimensions: [seq_len x num_haplotypes x num_classes].
 * \param sequences Input sequences, used to access class probabilities.
 * \param symbol_lookup Lookup table of symbol chars (bases) -> numeric ID of that symbol, to encode sequence bases into class IDs.
 * \param rstart Region start (start of the subsequence in seq). Zero-based.
 * \param rend Region end (end of the subsequence in seq). Non-inclusive.
 * \returns Phred-scaled quality for a given set of input sequences.
 */
float compute_quality_from_probs(
        const at::Tensor& probs_3D,  // Probabilities for a single sample (not batch).
        const std::span<std::string_view> sequences,
        const std::array<int32_t, 256>& symbol_lookup,
        const int64_t rstart,
        const int64_t rend) {
    if (std::size(probs_3D.sizes()) != 3) {
        throw std::runtime_error(
                "Tensor of probabilities given to compute_quality_from_probs is of wrong shape. "
                "Input "
                "shape: " +
                utils::tensor_shape_as_string(probs_3D) + ", but expected 3 dimensions.");
    }

    const int64_t num_haplotypes = probs_3D.size(1);

    if (std::ssize(sequences) != num_haplotypes) {
        throw std::runtime_error(
                "Number of haplotypes in the tensor differs from the number of "
                "sequences provided to compute_quality_from_probs. Tensor shape: " +
                utils::tensor_shape_as_string(probs_3D) +
                ", number of sequences: " + std::to_string(std::size(sequences)));
    }

    float total = 0.0f;
    for (int64_t hap_id = 0; hap_id < num_haplotypes; ++hap_id) {
        const float log_prob = compute_subseq_log_prob(probs_3D, sequences[hap_id], symbol_lookup,
                                                       rstart, rend, hap_id, false);
        total += log_prob;
    }
    total = phred(1.0f - std::exp(total), VCF_MAX_GQ_CAP);

    total = std::max(0.0f, total);

    return total;
}

/**
 * \brief Utility function to compute a quality score for the assertion that all haplotypes in a region are the reference sequence.
 *        This is the phred-scaled log-likelihood over the predictions that any allele is not reference.
 * \param probs_3D A 3D tensor of probabilities (output of inference) for a single sample (not batch).
 *                  Dimensions: [seq_len x num_haplotypes x num_classes].
 * \param ref_seq_with_gaps Input sequence, used to access class probabilities.
 * \param symbol_lookup Lookup table of symbol chars (bases) -> numeric ID of that symbol, to encode sequence bases into class IDs.
 * \param rstart Region start (start of the subsequence in seq). Zero-based.
 * \param rend Region end (end of the subsequence in seq). Non-inclusive.
 * \returns Phred-scaled quality for all haplotypes being reference given input haplotype predictions.
 */
float compute_ref_quality(
        const at::Tensor& probs_3D,  // Probabilities for a single sample (not batch).
        const std::string_view ref_seq_with_gaps,
        const std::array<int32_t, 256>& symbol_lookup,
        const int64_t rstart,
        const int64_t rend) {
    const int64_t num_haplotypes = probs_3D.size(1);

    std::vector<std::string_view> seqs;
    seqs.reserve(num_haplotypes);
    for (int64_t hap_id = 0; hap_id < num_haplotypes; ++hap_id) {
        seqs.emplace_back(ref_seq_with_gaps);
    }

    return compute_quality_from_probs(probs_3D, seqs, symbol_lookup, rstart, rend);
}

/**
 * \brief Utility function to compute a quality score for the assertion that the haplotypes in a region are the consensus sequences.
 *        This is the phred-scaled log-likelihood over the predictions that any predicted allele is wrong.
 * \param probs_3D A 3D tensor of probabilities (output of inference) for a single sample (not batch).
 *                  Dimensions: [seq_len x num_haplotypes x num_classes].
 * \param ref_seq_with_gaps Input sequence, used to access class probabilities.
 * \param symbol_lookup Lookup table of symbol chars (bases) -> numeric ID of that symbol, to encode sequence bases into class IDs.
 * \param rstart Region start (start of the subsequence in seq). Zero-based.
 * \param rend Region end (end of the subsequence in seq). Non-inclusive.
 * \returns Phred-scaled quality for the predicted haplotype sequences.
 */
float compute_consensus_quality(
        const at::Tensor& probs_3D,  // Probabilities for a single sample (not batch).
        const std::vector<ConsensusResult>& cons_seqs_with_gaps,
        const std::array<int32_t, 256>& symbol_lookup,
        const int64_t rstart,
        const int64_t rend) {
    std::vector<std::string_view> seqs;
    seqs.reserve(std::size(cons_seqs_with_gaps));
    for (const auto& val : cons_seqs_with_gaps) {
        seqs.emplace_back(val.seq);
    }

    return compute_quality_from_probs(probs_3D, seqs, symbol_lookup, rstart, rend);
}

Variant construct_variant(const std::string_view draft,
                          const std::vector<int64_t>& positions_major,
                          const std::vector<int64_t>& positions_minor,
                          const std::string_view ref_seq_with_gaps,
                          const std::vector<ConsensusResult>& cons_seqs_with_gaps,
                          const int32_t seq_id,
                          const int64_t rstart,
                          const int64_t rend,
                          const bool is_var,
                          const bool ambig_ref,
                          const bool normalize,
                          const at::Tensor& probs_3D,
                          const std::unordered_set<char>& symbol_set,
                          const std::array<int32_t, 256>& symbol_lookup) {
    const auto prepend_major_ref_base = [&positions_major, &positions_minor, &draft](Variant& var) {
        while ((var.rstart > 0) && (positions_minor[var.rstart] != 0)) {
            --var.rstart;
        }
        var.pos = positions_major[var.rstart];
        var.ref = draft[var.pos] + var.ref;
        const char base = draft[var.pos];
        for (std::string& alt : var.alts) {
            alt = base + std::move(alt);
        }
    };

    // Get the reference and the predictions for the variant stretch.
    const std::string_view var_ref_with_gaps(std::data(ref_seq_with_gaps) + rstart,
                                             static_cast<size_t>(rend - rstart));

    // Mutable ref and pred sequences - a ref base may be prepended later.
    std::string var_ref = remove_gaps(var_ref_with_gaps);

    std::vector<std::string> var_preds;
    for (int64_t hap_id = 0; hap_id < std::ssize(cons_seqs_with_gaps); ++hap_id) {
        const std::string_view var_pred_with_gaps(
                std::data(cons_seqs_with_gaps[hap_id].seq) + rstart,
                static_cast<size_t>(rend - rstart));
        var_preds.emplace_back(remove_gaps(var_pred_with_gaps));
    }

    if (is_var && std::all_of(std::cbegin(var_preds), std::cend(var_preds),
                              [&var_ref](const std::string_view val) { return val == var_ref; })) {
#ifdef DEBUG_NORMALIZE_VARIANT
        std::cerr << "[construct_variant] Returning empty variant, case 1!\n";
#endif
        return {};
    } else if (!ambig_ref && !is_subset_of_symbols(symbol_set, var_ref)) {
#ifdef DEBUG_NORMALIZE_VARIANT
        std::cerr << "[construct_variant] Returning empty variant, case 2!\n";
#endif
        return {};
    }

    Variant var{
            seq_id, positions_major[rstart],    var_ref, var_preds, "PASS", {},
            0.0f,   {{"GT", "1"}, {"GQ", "0"}}, rstart,  rend,
    };

    // Check if variant starts on insert, prepend previous ref base.
    if (positions_minor[var.rstart] != 0) {
#ifdef DEBUG_NORMALIZE_VARIANT
        std::cerr << "[construct_variant] Running prepend_major_ref_base.\n";
#endif
        prepend_major_ref_base(var);
    }

    if (normalize) {
        // Create a vector of views for normalize_variant.
        std::vector<std::string_view> cons_view;
        cons_view.reserve(std::size(cons_seqs_with_gaps));
        for (const ConsensusResult& val : cons_seqs_with_gaps) {
            cons_view.emplace_back(val.seq);
        }
#ifdef DEBUG_NORMALIZE_VARIANT
        std::cerr << "[construct_variant] Before normalization. var = " << var << '\n';
#endif
        var = normalize_variant(ref_seq_with_gaps, cons_view, positions_major, positions_minor,
                                symbol_set, var, ambig_ref);
#ifdef DEBUG_NORMALIZE_VARIANT
        std::cerr << "[construct_variant] After normalization.  var = " << var << '\n';
#endif
    }

    // If the ALT field is still is empty, set it to a '.'. This can happen when a deletion is
    // flanked by unreachable positions on both ends.
    const bool any_empty = std::any_of(std::cbegin(var.alts), std::cend(var.alts),
                                       [](const std::string_view val) { return std::empty(val); });
    if (std::empty(var.alts) || any_empty) {
        var.alts = {"."};
    }

    var.qual = round_float(compute_consensus_quality(probs_3D, cons_seqs_with_gaps, symbol_lookup,
                                                     var.rstart, var.rend),
                           3);

    return var;
}

std::vector<Variant> merge_sorted_variants(const std::vector<Variant>& variants,
                                           const bool merge_overlapping,
                                           const bool merge_adjacent,
                                           const std::string_view draft,
                                           const std::vector<int64_t>& positions_major,
                                           const std::vector<int64_t>& positions_minor,
                                           const std::string_view ref_seq_with_gaps,
                                           const std::vector<ConsensusResult>& cons_seqs_with_gaps,
                                           const bool ambig_ref,
                                           const bool normalize,
                                           const at::Tensor& probs_3D,
                                           const std::unordered_set<char>& symbol_set,
                                           const std::array<int32_t, 256>& symbol_lookup) {
    if (!merge_overlapping && !merge_adjacent) {
        return variants;
    }

    if (std::empty(variants)) {
        return {};
    }

    std::vector<Variant> filtered;
    int64_t furthest_rend = variants[0].rend;
    int64_t prev_i = 0;
    for (int64_t i = 1; i < std::ssize(variants); ++i) {
#ifdef DEBUG_NORMALIZE_VARIANT
        std::cerr << "[merge_sorted_variants] ----------------------------------\n";
#endif

        const Variant& v1 = variants[prev_i];
        const Variant& v2 = variants[i];
        const bool is_overlapping = (v2.rstart < furthest_rend) && (v2.rend >= v1.rstart);
        const bool is_adjacent = (v2.rstart == furthest_rend);

#ifdef DEBUG_NORMALIZE_VARIANT
        std::cerr << "[merge_sorted_variants] prev_i = " << prev_i << ", i = " << i
                  << ", is_overlapping = " << is_overlapping << ", is_adjacent = " << is_adjacent
                  << '\n';
        std::cerr << "[merge_sorted_variants] v1 = " << v1 << '\n';
        std::cerr << "[merge_sorted_variants] v2 = " << v2 << '\n';
#endif

        if ((merge_overlapping && is_overlapping) || (merge_adjacent && is_adjacent)) {
            furthest_rend = v2.rend;
#ifdef DEBUG_NORMALIZE_VARIANT
            std::cerr << "[merge_sorted_variants]   -> will merge, furthest_rend = "
                      << furthest_rend << '\n';
#endif
            continue;
        }

        if (furthest_rend != v1.rend) {
#ifdef DEBUG_NORMALIZE_VARIANT
            std::cerr << "[merge_sorted_variants] Constructing new_var.\n";
#endif

            Variant new_var = construct_variant(
                    draft, positions_major, positions_minor, ref_seq_with_gaps, cons_seqs_with_gaps,
                    variants[prev_i].seq_id, v1.rstart, furthest_rend, true, ambig_ref, normalize,
                    probs_3D, symbol_set, symbol_lookup);

#ifdef DEBUG_NORMALIZE_VARIANT
            std::cerr << "[merge_sorted_variants] new_var = " << new_var << '\n';
#endif

            if (is_valid(new_var)) {
                filtered.emplace_back(std::move(new_var));
            }
        } else {
            filtered.push_back(v1);
        }

        furthest_rend = v2.rend;
        prev_i = i;
    }
#ifdef DEBUG_NORMALIZE_VARIANT
    std::cerr << "[merge_sorted_variants] ----------------------------------\n";
#endif
    {  // Remaining.
        if (furthest_rend != variants[prev_i].rend) {
            Variant new_var = construct_variant(
                    draft, positions_major, positions_minor, ref_seq_with_gaps, cons_seqs_with_gaps,
                    variants[prev_i].seq_id, variants[prev_i].rstart, furthest_rend, true,
                    ambig_ref, normalize, probs_3D, symbol_set, symbol_lookup);

            if (is_valid(new_var)) {
                filtered.emplace_back(std::move(new_var));
            }
        } else {
            filtered.push_back(variants[prev_i]);
        }
    }
#ifdef DEBUG_NORMALIZE_VARIANT
    std::cerr << "[merge_sorted_variants] ==================================\n";
#endif

    return filtered;
}

std::tuple<bool, int64_t, int64_t> find_previous_ref_pos(
        const std::vector<int64_t>& positions_major,
        const std::vector<int64_t>& positions_minor,
        const int64_t rstart) {
    // Bounds check.
    if ((rstart <= 0) || (rstart >= std::ssize(positions_major))) {
        return {false, rstart, -1};
    }

    const int64_t ref_pos = positions_major[rstart];
    const int64_t prev_ref_pos = ref_pos - 1;

    // Can't go left.
    if (ref_pos <= 0) {
        return {false, rstart, ref_pos};
    }

    // Move.
    int64_t rpos = rstart;
    while ((rpos >= 0) &&
           ((positions_major[rpos] > prev_ref_pos) ||
            ((positions_major[rpos] == prev_ref_pos) && (positions_minor[rpos] != 0)))) {
        --rpos;
    }

    // Major not found, window begins with a minor position.
    if (rpos < 0) {
        return {false, rpos, ref_pos};
    }

    // Could not find the requested position.
    if ((positions_major[rpos] != prev_ref_pos) || (positions_minor[rpos] != 0)) {
        return {false, rpos, ref_pos};
    }

    // Found a valid position.
    return {true, rpos, prev_ref_pos};
}

std::tuple<bool, int64_t, int64_t> find_ref_pos(const std::vector<int64_t>& positions_major,
                                                const std::vector<int64_t>& positions_minor,
                                                const int64_t rstart,
                                                const int64_t requested_ref_pos) {
    const int64_t len = std::ssize(positions_major);

    // Bounds check.
    if ((requested_ref_pos < 0) || (rstart < 0) || (rstart >= len)) {
        return {false, -1, -1};
    }

    // Move. The `rpos` will be an inclusive end coordinate, so check the bounds.
    int64_t rpos = rstart;
    while ((rpos < len) &&
           ((positions_major[rpos] < requested_ref_pos) ||
            ((positions_major[rpos] == requested_ref_pos) && (positions_minor[rpos] != 0)))) {
        ++rpos;
    }

    // Major not found, window begins with a minor position.
    if (rpos >= len) {
        return {false, rpos, requested_ref_pos};
    }

    // Could not find the requested position.
    if ((positions_major[rpos] != requested_ref_pos) || (positions_minor[rpos] != 0)) {
        return {false, rpos, requested_ref_pos};
    }

    // The `rpos` coordinate here is an inclusive one.
    return {true, rpos, requested_ref_pos};
}

bool prepend_ref_base(Variant& var,
                      const std::string_view ref_with_gaps,
                      const std::vector<std::string_view>& cons_seqs_with_gaps,
                      const std::vector<int64_t>& positions_major,
                      const std::vector<int64_t>& positions_minor,
                      const std::unordered_set<char>& symbol_set,
                      const bool ambig_ref_allowed) {
    const auto [can_go_left, new_rstart, new_ref_pos] =
            find_previous_ref_pos(positions_major, positions_minor, var.rstart);

    if (!can_go_left) {
        return false;
    }

    // Prevent extension into ambiguous references.
    bool bases_valid = true;
    for (int64_t i = new_rstart; i < var.rstart; ++i) {
        if (!ambig_ref_allowed && (symbol_set.count(ref_with_gaps[i]) == 0)) {
            bases_valid = false;
            break;
        }
    }

    if (!bases_valid) {
        return false;
    }

    // Collect all prefixes for all consensus sequences and the ref, to prevent extension into variants.
    std::vector<std::string> prefixes;
    const int64_t span = var.rstart - new_rstart;
    prefixes.emplace_back(ref_with_gaps.substr(new_rstart, span));
    for (const std::string_view seq : cons_seqs_with_gaps) {
        prefixes.emplace_back(seq.substr(new_rstart, span));
    }

    // Create a set to count unique prefixes.
    const std::unordered_set<std::string> prefix_set(std::cbegin(prefixes), std::cend(prefixes));

    // Check if there is more than 1 unique prefix - if so, this is a variant and stop extension.
    if (std::size(prefix_set) > 1) {
        return false;
    }

    // Remove the deletions and prepend the prefix sequence.
    for (std::size_t i = 0; i < std::size(prefixes); ++i) {
        std::string& p = prefixes[i];
        p.erase(std::remove(std::begin(p), std::end(p), '*'), std::end(p));
    }

    // Finally, extend to the left.
    var.pos = positions_major[new_rstart];
    var.rstart = new_rstart;
    var.ref = prefixes[0] + var.ref;
    for (std::size_t i = 0; i < std::size(var.alts); ++i) {
        var.alts[i] = prefixes[i + 1] + var.alts[i];
    }

    return true;
}

bool append_ref_base(Variant& var,
                     const std::string_view ref_with_gaps,
                     const std::vector<std::string_view>& cons_seqs_with_gaps,
                     const std::vector<int64_t>& positions_major,
                     const std::vector<int64_t>& positions_minor,
                     const std::unordered_set<char>& symbol_set,
                     const bool ambig_ref_allowed) {
    // Do not rely on var.rend because bases might have been trimmed from
    // the back of the variant and var.rend left intact to track the entire region span.
    // Instead, find the desired reference coordinate for the new base,
    // then find the region pos (rpos) and add it.
    // Only the var.rstart should be considered valid.
    const int64_t next_ref_pos = var.pos + std::ssize(var.ref);

#ifdef DEBUG_NORMALIZE_VARIANT
    std::cerr << "[append_ref_base] On entry: var = " << var << '\n';
#endif
    // Search for the major position which matches the next_ref_pos,
    // starting from var.rstart.
    const auto [can_go_right, new_rend_inclusive, new_ref_pos] =
            find_ref_pos(positions_major, positions_minor, var.rstart, next_ref_pos);

    if (!can_go_right) {
#ifdef DEBUG_NORMALIZE_VARIANT
        std::cerr << "[append_ref_base] On return false 1: var = " << var << '\n';
#endif
        return false;
    }

    // Prevent extension into ambiguous references.
    bool bases_valid = var.rstart <= new_rend_inclusive;
    for (int64_t i = var.rstart; i < (new_rend_inclusive + 1); ++i) {
        if (!ambig_ref_allowed && (symbol_set.count(ref_with_gaps[i]) == 0)) {
            bases_valid = false;
            break;
        }
    }

    if (!bases_valid) {
#ifdef DEBUG_NORMALIZE_VARIANT
        std::cerr << "[append_ref_base] On return false 2: var = " << var << '\n';
#endif
        return false;
    }

    // Collect all prefixes for all consensus sequences and the ref.
    std::vector<char> suffixes;
    suffixes.emplace_back(ref_with_gaps[new_rend_inclusive]);
    for (const std::string_view seq : cons_seqs_with_gaps) {
        suffixes.emplace_back(seq[new_rend_inclusive]);
    }

    // Create a set to count unique prefixes.
    const std::unordered_set<char> suffix_set(std::cbegin(suffixes), std::cend(suffixes));

    // Check if there is more than 1 unique suffix - this is a variant then, stop extension.
    if (std::size(suffix_set) > 1) {
#ifdef DEBUG_NORMALIZE_VARIANT
        std::cerr << "[append_ref_base] On return false 3: var = " << var << '\n';
#endif
        return false;
    }

    // Finally, extend.
    const int64_t total_span = (new_rend_inclusive + 1) - var.rstart;
    var.ref = remove_gaps(ref_with_gaps.substr(var.rstart, total_span));
    for (int64_t i = 0; i < std::ssize(var.alts); ++i) {
        var.alts[i] = remove_gaps(cons_seqs_with_gaps[i].substr(var.rstart, total_span));
    }
    var.rend = new_rend_inclusive + 1;

#ifdef DEBUG_NORMALIZE_VARIANT
    std::cerr << "[append_ref_base] On return true: var = " << var << '\n';
#endif

    return true;
}

}  // namespace

Variant normalize_genotype(const Variant& var, const int32_t ploidy, const float min_qual) {
    Variant ret = var;

    if (std::ssize(var.alts) > ploidy) {
        spdlog::warn(
                "Number of alts ({}) is larger than ploidy ({})! Marking this variant for removal.",
                std::size(var.alts), ploidy);
        ret.alts.clear();
        return ret;
    }

    const int32_t gq = static_cast<int32_t>(std::round(var.qual));

    // This is a gVCF record.
    if (is_reference_record(var)) {
        const bool is_missing_call =
                (std::size(var.ref) == 1) && !utils::is_canonical_base(var.ref.front());
        const char genotype_allele = is_missing_call ? '.' : '0';

        if (std::empty(ret.alts)) {
            ret.alts = {"."};
        }

        const std::string gq_str = (genotype_allele == '.') ? "." : std::to_string(gq);

        if (is_missing_call) {
            ret.qual = -1.0f;
        }
        ret.genotype = {{"GT", make_reference_gt(ploidy, genotype_allele)}, {"GQ", gq_str}};
        ret.filter = ".";
        return ret;
    }

    // Create unique and sorted alts for the return variant.
    std::unordered_set<std::string> unique_alts;
    for (const std::string_view alt : var.alts) {
        if (alt != var.ref) {
            unique_alts.emplace(alt);
        }
    }
    ret.alts = std::vector<std::string>(std::cbegin(unique_alts), std::cend(unique_alts));
    std::stable_sort(std::begin(ret.alts), std::end(ret.alts));

    // Look-up table: alt -> numeric ID. +1 is because ref is the zeroth allele.
    std::unordered_map<std::string, int64_t> alt_dict;
    for (int64_t i = 0; i < std::ssize(ret.alts); ++i) {
        alt_dict[ret.alts[i]] = i + 1;
    }
    alt_dict[var.ref] = 0;

    std::vector<int32_t> alleles(std::size(var.alts));
    for (std::size_t i = 0; i < std::size(var.alts); ++i) {
        const auto it = alt_dict.find(var.alts[i]);
        if (it == std::cend(alt_dict)) {
            continue;
        }
        alleles[i] = it->second;
    }
    std::sort(std::begin(alleles), std::end(alleles));

    std::ostringstream oss_gt;
    for (std::size_t i = 0; i < std::size(alleles); ++i) {
        if (i > 0) {
            oss_gt << '/';
        }
        oss_gt << alleles[i];
    }

    ret.genotype = {{"GT", oss_gt.str()}, {"GQ", std::to_string(gq)}};
    ret.filter = (var.qual >= min_qual) ? "PASS" : "LowQual";

    return ret;
}

Variant collapse_to_haploid(const Variant& var, const bool require_hom) {
    Variant ret = var;

    // This is a gVCF record.
    if (is_reference_record(var)) {
        if (std::empty(ret.alts)) {
            ret.alts = {"."};
        }
        const bool is_missing_call =
                (std::size(ret.ref) == 1) && !utils::is_canonical_base(ret.ref.front());
        ret.filter = ".";
        if (is_missing_call) {
            ret.qual = -1.0f;
        }
        for (auto& val : ret.genotype) {
            if (val.first == "GT") {
                val.second = is_missing_call ? "." : "0";
            } else if (is_missing_call && (val.first == "GQ")) {
                val.second = ".";
            }
        }
        return ret;
    }

    const int64_t num_called_alts =
            std::count_if(std::cbegin(var.alts), std::cend(var.alts),
                          [](const std::string& alt) { return alt != "<*>"; });
    if (num_called_alts > 1) {
        spdlog::debug("Discarding variant with more than one alt in haploid region ({}, {}).",
                      var.seq_id, var.pos);
        ret.alts.clear();
        ret.filter = ".";
        for (auto& it_gt : ret.genotype) {
            if (it_gt.first == "GT") {
                it_gt.second = "0";
                break;
            }
        }
        return ret;
    }

    bool is_homozygous = true;
    std::vector<std::pair<std::string, std::string>>::iterator it_gt;
    for (it_gt = ret.genotype.begin(); it_gt != ret.genotype.end(); ++it_gt) {
        if (it_gt->first == "GT") {
            if (it_gt->second.find("0") != std::string::npos) {
                is_homozygous = false;
            }
            break;
        }
    }
    if (it_gt == ret.genotype.end()) {
        spdlog::warn("Failed to find genotype record for variant at ({}, {})", ret.seq_id, ret.pos);
        ret.alts.clear();
    }
    if (is_homozygous || !require_hom) {
        it_gt->second = "1";
    } else {
        ret.alts.clear();
        it_gt->second = "0";
        spdlog::debug("Discarding heterozygous variant ({}, {}) in haploid region.", ret.seq_id,
                      ret.pos);
    }
    return ret;
}

Variant normalize_variant(const std::string_view ref_with_gaps,
                          const std::vector<std::string_view>& cons_seqs_with_gaps,
                          const std::vector<int64_t>& positions_major,
                          const std::vector<int64_t>& positions_minor,
                          const std::unordered_set<char>& symbol_set,
                          const Variant& variant,
                          const bool ambig_ref) {
    const bool all_same_as_ref =
            std::all_of(std::cbegin(variant.alts), std::cend(variant.alts),
                        [&variant](const std::string_view s) { return s == variant.ref; });

    if (all_same_as_ref) {
        return variant;
    }

    const auto trim_start = [](Variant& var, const bool rev) {
        // Get the sequences and reverse if needed.
        std::vector<std::string> seqs{var.ref};
        seqs.insert(std::end(seqs), std::cbegin(var.alts), std::cend(var.alts));
        if (rev) {
            for (auto& seq : seqs) {
                std::reverse(std::begin(seq), std::end(seq));
            }
        }

        // Sanity check.
        if (std::size(seqs) < 2) {
            return;
        }

        const int64_t min_len = std::ssize(
                *std::min_element(std::cbegin(seqs), std::cend(seqs),
                                  [](const std::string_view a, const std::string_view b) {
                                      return std::ssize(a) < std::ssize(b);
                                  }));

        // Never trim the last base.
        int64_t start_pos = 0;
        for (int64_t i = 0; i < (min_len - 1); ++i) {
            bool bases_same = true;
            for (int64_t j = 1; j < std::ssize(seqs); ++j) {
                if (seqs[j][i] != seqs[0][i]) {
                    bases_same = false;
                    break;
                }
            }
            if (!bases_same) {
                break;
            }
            ++start_pos;
        }

        // Trim.
        if (start_pos > 0) {
            for (auto& seq : seqs) {
                seq = seq.substr(start_pos);
            }
        }
        // Reverse if needed.
        if (rev) {
            for (auto& seq : seqs) {
                std::reverse(std::begin(seq), std::end(seq));
            }
            start_pos = 0;
        }
        // Assign.
        var.ref = seqs[0];
        var.alts = std::vector<std::string>(std::cbegin(seqs) + 1, std::cend(seqs));
        var.pos += start_pos;
    };

    const auto trim_end_and_align = [&ref_with_gaps, &cons_seqs_with_gaps, &positions_major,
                                     &positions_minor, &symbol_set, &ambig_ref](Variant& var) {
        const auto reset_var = [](const Variant& v) {
#ifdef DEBUG_NORMALIZE_VARIANT
            std::cerr << "[normalize_variant]    [trim_end_and_align] Resetting var.\n";
#endif
            std::vector<std::string> seqs{v.ref};
            seqs.insert(std::end(seqs), std::cbegin(v.alts), std::cend(v.alts));
            return std::make_pair(v, seqs);
        };

        std::vector<std::string> seqs{var.ref};
        seqs.insert(std::end(seqs), std::cbegin(var.alts), std::cend(var.alts));

        bool changed = true;
        while (changed) {
#ifdef DEBUG_NORMALIZE_VARIANT
            std::cerr << "[normalize_variant]    [trim_end_and_align] while (changed). var = "
                      << var << '\n';
#endif

            changed = false;

            // Keep a copy if we need to bail.
            const Variant var_before_change = var;

            const bool all_non_empty =
                    std::all_of(std::cbegin(seqs), std::cend(seqs),
                                [](const std::string_view s) { return !std::empty(s); });

            // Trim the last base if identical (right trim).
            if (all_non_empty) {
                // Check if the last base is identical in all seqs.
                bool all_same = true;
                for (int64_t i = 1; i < std::ssize(seqs); ++i) {
                    if (seqs[i].back() != seqs[0].back()) {
                        all_same = false;
                        break;
                    }
                }

                // Trim right.
                if (all_same) {
                    for (auto& seq : seqs) {
                        assert(!seq.empty());
                        seq.pop_back();
                    }
                    changed = true;
                    var.ref = seqs[0];
                    var.alts = std::vector<std::string>(std::cbegin(seqs) + 1, std::cend(seqs));
                }
            }

#ifdef DEBUG_NORMALIZE_VARIANT
            std::cerr << "[normalize_variant]    [trim_end_and_align]    -> After all_non_empty. "
                         "var = "
                      << var << '\n';
#endif

            const bool any_empty =
                    std::any_of(std::cbegin(seqs), std::cend(seqs),
                                [](const std::string_view s) { return std::empty(s); });

            // Extend. Prepend/append a base if any seq is empty.
            if (any_empty) {
                bool used_right_extend = false;
                changed = prepend_ref_base(var, ref_with_gaps, cons_seqs_with_gaps, positions_major,
                                           positions_minor, symbol_set, ambig_ref);

                if (!changed) {
                    // std::tie(var, seqs) = reset_var(var_before_change);
                    changed = append_ref_base(var, ref_with_gaps, cons_seqs_with_gaps,
                                              positions_major, positions_minor, symbol_set,
                                              ambig_ref);
                    used_right_extend = true;
                }

                if (changed) {
                    // Reset the seqs because the extension functions don't care about them.
                    seqs = {var.ref};
                    seqs.insert(std::end(seqs), std::cbegin(var.alts), std::cend(var.alts));
                } else {
                    // Reset and bail.
                    std::tie(var, seqs) = reset_var(var_before_change);
                    break;
                }

                if (used_right_extend) {
                    // Stop normalization because we can't go left any further and we would just
                    // be trimming and adding bases indefinitely.
                    break;
                }
            }

#ifdef DEBUG_NORMALIZE_VARIANT
            std::cerr << "[normalize_variant]    [trim_end_and_align]    -> After any_empty. var = "
                      << var << '\n';
#endif
        }
        var.ref = seqs[0];
        var.alts = std::vector<std::string>(std::cbegin(seqs) + 1, std::cend(seqs));

#ifdef DEBUG_NORMALIZE_VARIANT
        std::cerr << "[normalize_variant]    [trim_end_and_align] After while. var = " << var
                  << '\n';
#endif
    };

    Variant ret = variant;

    // Normalize the start of the variant. For example, if the input variant represents a region like this:
    // - POS  :      43499195    43499196
    //               v           v
    // - REF  : CCTAG************TTATTATT
    // - HAP 0: CCTAG*********TT**T*TTATT
    // - HAP 1: CCTAG*********T*AT*ATTATT
    // - VAR  : 0000011111111111111100000
    // - MARK :      ^
    //
    // it is possible that the input variant.pos was set to the pos_major of the beginning of the variant
    // (in this case, on a minor position which does not contain a reference base).
    //
    // Here we try to move the start position to the first major base left, then right.
    int32_t new_rstart = ret.rstart;
    while ((new_rstart > 0) && (positions_minor[new_rstart] != 0)) {
        --new_rstart;
    }
    if (positions_minor[new_rstart] != 0) {
        new_rstart = ret.rstart + 1;
        while ((new_rstart < ret.rend) && (positions_minor[new_rstart] != 0)) {
            ++new_rstart;
        }
    }
    if (new_rstart >= ret.rend) {
        return {};
    }
    if (new_rstart != ret.rstart) {
        ret.rstart = new_rstart;
        ret.pos = positions_major[ret.rstart];
        ret.ref = remove_gaps(ref_with_gaps.substr(ret.rstart, ret.rend - ret.rstart));
        ret.alts = {};
        for (const auto& s : cons_seqs_with_gaps) {
            ret.alts.emplace_back(remove_gaps(s.substr(ret.rstart, ret.rend - ret.rstart)));
        }
    }

#ifdef DEBUG_NORMALIZE_VARIANT
    std::cerr << "[normalize_variant] Entry variant: " << variant << '\n';
#endif

    if (std::empty(ref_with_gaps)) {
#ifdef DEBUG_NORMALIZE_VARIANT
        std::cerr << "[normalize_variant] Before trim_start 1: " << ret << '\n';
#endif
        trim_start(ret, true);
    } else {
#ifdef DEBUG_NORMALIZE_VARIANT
        std::cerr << "[normalize_variant] Before trim_end_and_align: " << ret << '\n';
#endif
        trim_end_and_align(ret);
    }

#ifdef DEBUG_NORMALIZE_VARIANT
    std::cerr << "[normalize_variant] Before trim_start 2:" << ret << '\n';
#endif

    trim_start(ret, false);

#ifdef DEBUG_NORMALIZE_VARIANT
    std::cerr << "[normalize_variant] Final variant: " << ret << '\n';
#endif

    return ret;
}

std::vector<Variant> general_decode_variants(
        const DecoderBase& decoder,
        const int32_t seq_id,
        const std::vector<int64_t>& positions_major,
        const std::vector<int64_t>& positions_minor,
        const at::Tensor& probs,  // Probabilities for a single sample (not batch).
        const std::string_view draft,
        const float pass_min_qual,
        const bool ambig_ref,
        const bool return_all,
        const bool normalize,
        const bool merge_overlapping,
        const bool merge_adjacent,
        const std::span<const std::pair<int64_t, float>> gvcf_reference_block_gq_margins) {
    const int64_t num_columns = std::ssize(positions_major);

    // Validate inputs.
    if (seq_id < 0) {
        throw std::runtime_error("Sequence ID is negative in general_decode_variants: " +
                                 std::to_string(seq_id));
    }
    if (std::size(positions_major) != std::size(positions_minor)) {
        std::ostringstream oss;
        oss << "Size of positions_major (" << std::size(positions_major)
            << ") and positions_minor (" << std::size(positions_minor)
            << ") differs in general_decode_variants.";
        throw std::runtime_error(oss.str());
    }
    if (!probs.defined()) {
        throw std::runtime_error(
                "The tensor of probabilities is not defined in general_decode_variants.");
    }
    if (probs.size(0) != num_columns) {
        std::ostringstream oss;
        oss << "Input probs tensor is of incorrect size. probs.size = " << probs.size(0)
            << ", num_columns = " << num_columns;
        throw std::runtime_error(oss.str());
    }
    if ((std::size(probs.sizes()) < 2) || (std::size(probs.sizes()) > 3)) {
        throw std::runtime_error("Input probs tensor is of unexpected size. Tensor dimensions: " +
                                 std::to_string(probs.sizes().size()) + ", expected 2 or 3.");
    }

    // No work to do.
    if (std::empty(positions_major)) {
        return {};
    }

    // Systematize the input probabilities to support the legacy case (shape [num_pos x num_classes])
    // and the new case (shape [num_pos x num_haps x num_classes]).
    const at::Tensor probs_3D = (std::size(probs.sizes()) == 2) ? probs.unsqueeze(1) : probs;

    // Symbols and lookup.
    const std::string symbols = decoder.get_label_scheme_symbols();
    const std::unordered_set<char> symbol_set(std::cbegin(symbols), std::cend(symbols));
    const std::array<int32_t, 256> symbol_lookup = create_symbol_lookup(symbols);

    // Get raw probability data.
    const size_t batch_size = 1;
    const size_t seq_len = std::size(positions_major);
    const size_t num_haplotypes = static_cast<size_t>(probs_3D.size(1));
    const size_t num_classes = std::size(symbols);
    const std::span<const float> raw_probs_data(
            probs_3D.data_ptr<float>(), batch_size * seq_len * num_haplotypes * num_classes);

    // Consensus sequences.
    const std::vector<std::vector<ConsensusResult>> cons_seqs_with_gaps_all =
            decode_batch_bases_impl(symbols, raw_probs_data, batch_size, seq_len, num_haplotypes,
                                    num_classes);

    if (std::size(cons_seqs_with_gaps_all) != 1) {
        spdlog::warn(
                "Unexpected number of decoded samples in general_decode_variants. Output size: {}, "
                "expected 1. Returning zero variants for this sample.",
                std::size(cons_seqs_with_gaps_all));
        return {};
    }

    const std::vector<ConsensusResult>& cons_seqs_with_gaps = cons_seqs_with_gaps_all.front();

    // Draft sequence with gaps.
    const std::string ref_seq_with_gaps =
            extract_draft_with_gaps(draft, positions_major, positions_minor);

    // Candidate variant positions.
    const std::vector<bool> is_variant =
            find_polyploid_variants(positions_minor, ref_seq_with_gaps, cons_seqs_with_gaps,
                                    (!ambig_ref) ? std::optional(symbol_set) : std::nullopt);

    const std::vector<std::tuple<int64_t, int64_t, bool>> runs =
            dorado::run_length_encode(is_variant);

#ifdef DEBUG_VARIANT_REGIONS
    std::vector<std::string_view> cons_view;
    cons_view.reserve(std::size(cons_seqs_with_gaps));
    for (const ConsensusResult& val : cons_seqs_with_gaps) {
        cons_view.emplace_back(val.seq);
    }
#endif

#ifdef DEBUG_VC_DATA_DUMP
    {
        std::vector<std::string_view> cons_view2;
        cons_view2.reserve(std::size(cons_seqs_with_gaps));
        for (const ConsensusResult& val : cons_seqs_with_gaps) {
            cons_view2.emplace_back(val.seq);
        }
        std::cerr << "[general_decode_variants] seq_id = " << seq_id
                  << ", columns = " << std::size(positions_major) << '\n';
        print_slice(std::cerr, ref_seq_with_gaps, cons_view2, positions_major, positions_minor,
                    is_variant, 0, -1, 0, -1);
    }
#endif

    // Extract variants.
    std::vector<Variant> variants;
    for (const auto& [rstart, rend, is_var] : runs) {
        // Skip non-variants.
        if (!is_var) {
            continue;
        }

#ifdef DEBUG_VARIANT_REGIONS
        {
            std::cerr << "[general_decode_variants] ---------------------------------------\n";
        }
#endif

        Variant var = construct_variant(draft, positions_major, positions_minor, ref_seq_with_gaps,
                                        cons_seqs_with_gaps, seq_id, rstart, rend, is_var,
                                        ambig_ref, normalize, probs_3D, symbol_set, symbol_lookup);

#ifdef DEBUG_VARIANT_REGIONS
        {
            std::cerr << "[general_decode_variants] region: rstart = " << rstart
                      << ", rend = " << rend << ", is_var = " << is_var << '\n';
            std::cerr << "[general_decode_variants slice] var = " << var << '\n';
            const int64_t s = std::max<int64_t>(0, var.rstart - 5);
            const int64_t e = std::min(std::ssize(positions_major), var.rend + 5);
            print_slice(std::cerr, ref_seq_with_gaps, cons_view, positions_major, positions_minor,
                        is_variant, s, e, var.rstart, var.rend);
            std::cerr << '\n';
        }
#endif

        if (!is_valid(var)) {
            continue;
        }

        variants.emplace_back(std::move(var));
    }
#ifdef DEBUG_VARIANT_REGIONS
    {
        std::cerr << "[general_decode_variants] =======================================\n";
    }
#endif

#ifdef DEBUG_NORMALIZE_VARIANT
    std::cerr << "Collected variants (" << std::size(variants) << "):\n";
    for (size_t i = 0; i < std::size(variants); ++i) {
        std::cerr << "    [i = " << i << "] var: " << variants[i] << '\n';
    }
    std::cerr << '\n';
#endif

    if (merge_overlapping || merge_adjacent) {
        std::sort(std::begin(variants), std::end(variants));

        variants = merge_sorted_variants(variants, merge_overlapping, merge_adjacent, draft,
                                         positions_major, positions_minor, ref_seq_with_gaps,
                                         cons_seqs_with_gaps, ambig_ref, normalize, probs_3D,
                                         symbol_set, symbol_lookup);
    }

#ifdef DEBUG_NORMALIZE_VARIANT
    std::cerr << "Merged variants (" << std::size(variants) << "):\n";
    for (size_t i = 0; i < std::size(variants); ++i) {
        std::cerr << "    [i = " << i << "] var: " << variants[i] << '\n';
    }
    std::cerr << '\n';
#endif

    if (return_all) {
        std::sort(std::begin(variants), std::end(variants));

        const std::size_t num_existing_variants = std::size(variants);
        std::size_t variant_idx = 0;

        const auto find_next_variant = [&](std::size_t idx, const int64_t pos) {
            while ((idx < num_existing_variants) &&
                   variant_ends_before_position(variants[idx], seq_id, pos)) {
                ++idx;
            }
            return idx;
        };

        const auto get_reference_gq = [&](const bool is_callable, const int64_t start,
                                          const int64_t end) {
            if (!is_callable) {
                return 0.0f;
            }
            return round_float(
                    compute_ref_quality(probs_3D, ref_seq_with_gaps, symbol_lookup, start, end), 3);
        };

        int64_t i = 0;
        while (i < std::ssize(positions_major)) {
            // Skip non-reference positions.
            if (positions_minor[i] != 0) {
                ++i;
                continue;
            }

            const int64_t pos = positions_major[i];
            variant_idx = find_next_variant(variant_idx, pos);
            if ((variant_idx < num_existing_variants) &&
                variant_covers_position(variants[variant_idx], seq_id, pos)) {
                ++i;
                continue;
            }

            const int64_t block_start_idx = i;
            const int64_t block_start_pos = pos;
            int64_t block_end_idx = i + 1;
            int64_t block_end_pos = pos + 1;
            const bool is_callable_block = utils::is_canonical_base(draft[block_start_pos]);

            const float first_gq =
                    get_reference_gq(is_callable_block, block_start_idx, block_end_idx);
            const float gq_margin =
                    get_gvcf_reference_record_gq_margin(first_gq, gvcf_reference_block_gq_margins);
            float min_gq = first_gq;

            ++i;
            while (i < std::ssize(positions_major)) {
                if (positions_minor[i] != 0) {
                    ++i;
                    continue;
                }

                const int64_t current_pos = positions_major[i];
                variant_idx = find_next_variant(variant_idx, current_pos);
                if ((current_pos != block_end_pos) ||
                    ((variant_idx < num_existing_variants) &&
                     variant_covers_position(variants[variant_idx], seq_id, current_pos))) {
                    break;
                }

                if (utils::is_canonical_base(draft[current_pos]) != is_callable_block) {
                    break;
                }
                const float current_gq = get_reference_gq(is_callable_block, i, i + 1);
                if (std::fabs(current_gq - first_gq) > gq_margin) {
                    break;
                }

                min_gq = std::min(min_gq, current_gq);
                block_end_idx = i + 1;
                block_end_pos = current_pos + 1;
                ++i;
            }

            const std::string ref(1, draft[block_start_pos]);
            Variant var{
                    seq_id,
                    block_start_pos,
                    ref,
                    {"."},
                    ".",
                    {},
                    min_gq,
                    {{"GT", "0"}, {"GQ", "0"}},
                    block_start_idx,
                    block_end_idx,
            };

            set_gvcf_reference_block_end(var, block_end_pos);

            variants.emplace_back(std::move(var));
        }
    }

    std::sort(std::begin(variants), std::end(variants));

    // Normalize variants.
    std::vector<Variant> normalized_variants;
    normalized_variants.reserve(std::size(variants));
    for (const Variant& var : variants) {
        Variant new_var = normalize_genotype(var, num_haplotypes, pass_min_qual);
        if (const auto it = new_var.info.find("END");
            (new_var.filter == ".") && (it != std::cend(new_var.info))) {
            set_gvcf_reference_block_end(new_var, std::stoll(it->second));
        }

        // Sanity check.
        if (is_valid(new_var)) {
            normalized_variants.emplace_back(std::move(new_var));
        }
    }

    return normalized_variants;
}

}  // namespace dorado::secondary
