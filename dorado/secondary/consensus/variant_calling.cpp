#include "secondary/consensus/variant_calling.h"

#include "secondary/consensus/consensus_result.h"
#include "secondary/consensus/consensus_utils.h"
#include "secondary/features/encoder_utils.h"
#include "secondary/features/medaka_read_matrix_constants.h"
#include "torch_utils/tensor_utils.h"
#include "utils/container_utils.h"
#include "utils/log_utils.h"
#include "utils/rle.h"
#include "utils/sequence_utils.h"
#include "utils/stl_utils.h"
#include "utils/string_utils.h"

#include <spdlog/fmt/bundled/format.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

// #define DEBUG_VARIANT_REGIONS
// #define DEBUG_NORMALIZE_VARIANT

namespace dorado::secondary {

namespace {

constexpr float NEGINF = -std::numeric_limits<float>::infinity();
constexpr float EPS = 1.e-10f;
constexpr float LOG_TO_PHRED_FACTOR = -10.0f / std::numbers::ln10_v<float>;

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
    float ret = std::min(q, cap);
    if (ret <= 0.0f) {
        ret = 0.0f;
    }
    return ret;
}

/**
 * \brief Compute phred quality from log-probability, with a max QV cap.
 */
float phred_from_log(float logerr, const float cap) {
    logerr = std::clamp(logerr, cap / LOG_TO_PHRED_FACTOR, 0.0f);
    const float q = logerr * LOG_TO_PHRED_FACTOR;
    return std::min(q, cap);
}

float logaddexp(const float lhs, const float rhs) {
    if (lhs == NEGINF) {
        return rhs;
    }
    if (rhs == NEGINF) {
        return lhs;
    }
    const float hi = std::max(lhs, rhs);
    const float lo = std::min(lhs, rhs);
    return hi + std::log1p(std::exp(lo - hi));
}

/**
 * \brief Create a lookup table of valid symbols.
 */
std::array<int32_t, 256> create_symbol_lookup(const std::string_view symbols) {
    std::array<int32_t, 256> ret;
    std::fill(std::begin(ret), std::end(ret), -1);
    for (int32_t i = 0; i < std::ssize(symbols); ++i) {
        ret[static_cast<uint8_t>(symbols[i])] = i;
    }
    return ret;
}

/**
 * \brief Collect unique allele sequences.
 * \param ref Gap-compressed reference sequence.
 * \param preds Gap-compressed consensus sequence for each haplotype.
 * \returns Map of unique alleles to index, reference=0 followed by sorted alts.
 */
std::vector<std::string> construct_alleles_lookup(const std::string_view ref,
                                                  const std::span<const std::string> preds) {
    // Create unique and sorted alts for the return variant.
    std::set<std::string> unique_alts;
    for (const std::string& alt : preds) {
        if (alt != ref) {
            unique_alts.emplace(alt);
        }
    }
    std::vector<std::string> alleles_lookup;
    alleles_lookup.reserve(std::size(unique_alts) + 1);
    alleles_lookup.emplace_back(std::string(ref));
    alleles_lookup.insert(std::end(alleles_lookup), std::cbegin(unique_alts),
                          std::cend(unique_alts));
    return alleles_lookup;
}

std::vector<std::string> construct_alts_out(const std::span<const std::string> alleles) {
    if (std::empty(alleles)) {
        return {};
    }
    if (std::ssize(alleles) == 1) {
        return {"."};
    }
    std::vector<std::string> alts;
    alts.reserve(std::size(alleles) - 1);
    for (size_t i = 1; i < std::size(alleles); ++i) {
        alts.emplace_back(std::string(alleles[i]));
    }
    return alts;
}

std::vector<int32_t> make_class_indices(const std::string_view seq,
                                        const std::array<int32_t, 256>& symbol_lookup,
                                        const int64_t num_classes,
                                        const int64_t blank,
                                        const bool substitute_n,
                                        const bool keep_blank) {
    std::vector<int32_t> class_idxs;
    class_idxs.reserve(std::size(seq));
    for (const char c : seq) {
        const char base = (substitute_n && (c == 'N')) ? '*' : c;
        const int32_t class_id = symbol_lookup[static_cast<uint8_t>(base)];
        if ((class_id < 0) || (class_id >= num_classes)) {
            throw std::runtime_error("Sequence contains symbol with invalid class index.");
        }
        if (keep_blank || (class_id != blank)) {
            class_idxs.emplace_back(class_id);
        }
    }
    return class_idxs;
}

std::vector<int32_t> get_gt_indices(const std::span<const std::string> preds,
                                    const std::span<const std::string> alleles_lookup) {
    std::vector<int32_t> indices;
    indices.reserve(std::size(preds));
    for (const auto& pred : preds) {
        const auto idx = std::find(std::cbegin(alleles_lookup), std::cend(alleles_lookup), pred);
        if (idx == std::cend(alleles_lookup)) {
            throw std::runtime_error{"Allele " + pred + " not in unique alleles lookup"};
        };
        indices.emplace_back(std::distance(std::cbegin(alleles_lookup), idx));
    }
    return indices;
}

/**
 * \brief Calculate the number of basecalls supporting each of the alleles.
 *        It is assumed that alleles has been deduplicated. If any duplicated entries are
 *        present in alleles, then matches will only be counted for the first matching allele.
 * \param alleles Gap-compressed allele strings.
 * \param rstart Search start coordinate in the cumulative position dimension of basecalls.
 * \param rend Search end coordinate in the cumulative position dimension of basecalls.
 * \param basecalls Basecall tensor taken from a VariantCallingSample.
 * \returns Count of basecall reads that match each allele, formatted as a comma-delimited string.
 */
std::string calculate_basecall_allele_depths(const std::vector<std::string>& alleles,
                                             const int64_t rstart,
                                             const int64_t rend,
                                             const at::Tensor& basecalls) {
    if ((!basecalls.defined()) || (basecalls.dim() != 2)) {
        DORADO_LOG(debug,
                   "[add_basecall_allele_depths] Basecall features have incorrect shape, cannot "
                   "calculate allele depth.");
        return {};
    }
    if (basecalls.dtype() != at::kChar) {
        DORADO_LOG(debug,
                   "[add_basecall_allele_depths] Basecall features have unexpected dtype: {}",
                   at::toString(basecalls.dtype()));
        return {};
    }
    if (!basecalls.is_contiguous()) {
        DORADO_LOG(debug, "[add_basecall_allele_depths] Basecall features are not contiguous");
        return {};
    }
    const int64_t num_positions = basecalls.size(0);
    if ((rstart < 0) || (rend <= rstart) || (rstart > num_positions) || (rend > num_positions)) {
        return {};
    }

    const int64_t num_reads = basecalls.size(1);
    const int64_t width = rend - rstart;
    const size_t start_idx = rstart * num_reads;
    const std::span<const int8_t> data(basecalls.data_ptr<int8_t>(),
                                       static_cast<size_t>(num_positions * num_reads));

    std::vector<std::vector<int8_t>> encoded_alleles;
    encoded_alleles.reserve(std::size(alleles));
    for (const std::string_view allele : alleles) {
        encoded_alleles.emplace_back(encode_sequence(allele));
    }
    std::vector<int64_t> counts(std::size(alleles), 0);

    for (int64_t read = 0; read < num_reads; ++read) {
        size_t idx = start_idx + read;
        std::vector<int8_t> bases;
        bases.reserve(width);
        for (int64_t pos = 0; pos < width; ++pos, idx += num_reads) {
            const int8_t base = data[idx];
            // Retain missing data 0 values so that reads that don't span the interval
            // are not counted - 0's should never occur in the target sequences.
            if (base != DEL_VAL) {
                bases.emplace_back(base);
            }
        }
        const auto it_alleles =
                std::find(std::cbegin(encoded_alleles), std::cend(encoded_alleles), bases);
        if (it_alleles != std::cend(encoded_alleles)) {
            ++counts[std::distance(std::cbegin(encoded_alleles), it_alleles)];
        }
    }
    return utils::print_container_as_string(counts, ",", false);
}

/**
 * \brief Utility function to compute a log probability of a subsequence occurring for
 *          a specified haplotype.
 * \param probs_3D A 3D tensor of probabilities (output of inference) for a single sample (not batch).
 *                  Dimensions: [total_seq_len x num_haplotypes x num_classes].
 * \param seq_class_idxs Encoded sequence class indices, including blanks for explicit path scoring.
 * \param rstart Region start (start of the subsequence in seq). Zero-based.
 * \param rend Region end (end of the subsequence in seq). Non-inclusive.
 * \param hap_id Haplotype ID, needed to access class probabilities for that particular haplotype in the probs_3D tensor.
 * \returns A single value with the log probability of that subsequence occurring for the specified haplotype, given the class probabilities obtained through inference.
 */
float compute_subseq_log_prob(
        const at::Tensor& probs_3D,  // Tensor sliced to the [rstart, rend] range
        const std::span<const int32_t> seq_class_idxs,
        const int64_t rstart,
        const int64_t rend,
        const int64_t hap_id) {
    if (std::size(probs_3D.sizes()) != 3) {
        throw std::runtime_error(
                "Tensor of probabilities given to compute_subseq_quality is of wrong shape. Input "
                "shape: " +
                utils::tensor_shape_as_string(probs_3D) + ", but expected 3 dimensions.");
    }

    const int64_t num_pos = probs_3D.size(0);
    const int64_t num_haplotypes = probs_3D.size(1);
    const int64_t num_classes = probs_3D.size(2);

    if ((rstart < 0) || (rstart > rend) || (rstart >= num_pos) || (rend > num_pos)) {
        throw std::runtime_error{"rstart=" + std::to_string(rstart) +
                                 ",rend=" + std::to_string(rend) +
                                 " coordinates are out of range for tensor with " +
                                 std::to_string(num_pos) + " positions"};
    }
    if (hap_id >= num_haplotypes) {
        throw std::runtime_error{
                "hap_id=" + std::to_string(hap_id) +
                " is out of range for tensor with ploidy=" + std::to_string(num_haplotypes)};
    }
    if (std::any_of(std::cbegin(seq_class_idxs), std::cend(seq_class_idxs),
                    [&num_classes](const int32_t t) { return t >= num_classes; })) {
        throw std::runtime_error{
                "seq_class_idxs=" + utils::print_container_as_string(seq_class_idxs, ",", true) +
                " contains out-of-range index for tensor with " + std::to_string(num_haplotypes) +
                " classes"};
    }

    if ((rend - rstart) != std::ssize(seq_class_idxs)) {
        throw std::runtime_error("Target length " + std::to_string(std::size(seq_class_idxs)) +
                                 " does not span interval " + std::to_string(rstart) + "-" +
                                 std::to_string(rend));
    }

    const std::span<const float> data(probs_3D.data_ptr<float>(),
                                      static_cast<size_t>(probs_3D.numel()));

    if (std::empty(data)) {
        return 0.0f;
    }

    float total = 0.0f;
    for (int64_t pos = 0; pos < rend - rstart; ++pos) {
        const int64_t idx = (rstart + pos) * num_haplotypes * num_classes + hap_id * num_classes +
                            seq_class_idxs[static_cast<size_t>(pos)];
        const float prob = std::max(EPS, data[idx]);
        const float log_prob = std::log(prob);

        total += log_prob;
    }

    return total;
}

/**
 * \brief Utility function to compute a log probability of a gap-compressed sequence.
 * \param probs_3D A 3D tensor of probabilities (output of inference) for a single sample (not batch).
 *                  Dimensions: [total_seq_len x num_haplotypes x num_classes].
 * \param seq_class_idxs Encoded sequence class indices, excluding blanks.
 * \param blank_class_idx blank class index.
 * \param rstart Region start (start of the subsequence in seq). Zero-based.
 * \param rend Region end (end of the subsequence in seq). Non-inclusive.
 * \param hap_id Haplotype ID, needed to access class probabilities for that particular haplotype in the probs_3D tensor.
 * \returns A single value with the log probability of that subsequence occurring for the specified haplotype, given the class probabilities obtained through inference.
 */
float compute_subseq_log_prob_dp(
        const at::Tensor& probs_3D,  // Tensor sliced to the [rstart, rend] range
        const std::span<const int32_t> seq_class_idxs,
        const int64_t blank_class_idx,
        const int64_t rstart,
        const int64_t rend,
        const int64_t hap_id) {
    if (std::size(probs_3D.sizes()) != 3) {
        throw std::runtime_error(
                "Tensor of probabilities given to compute_subseq_quality is of wrong shape. Input "
                "shape: " +
                utils::tensor_shape_as_string(probs_3D) + ", but expected 3 dimensions.");
    }

    const int64_t num_pos = probs_3D.size(0);
    const int64_t num_haplotypes = probs_3D.size(1);
    const int64_t num_classes = probs_3D.size(2);
    const int64_t rlen = rend - rstart;

    if ((rstart < 0) || (rstart > rend) || (rstart >= num_pos) || (rend > num_pos)) {
        throw std::runtime_error{"rstart=" + std::to_string(rstart) +
                                 ",rend=" + std::to_string(rend) +
                                 " coordinates are out of range for tensor with " +
                                 std::to_string(num_pos) + " positions"};
    }
    if (hap_id >= num_haplotypes) {
        throw std::runtime_error{
                "hap_id=" + std::to_string(hap_id) +
                " is out of range for tensor with ploidy=" + std::to_string(num_haplotypes)};
    }
    if (std::any_of(std::cbegin(seq_class_idxs), std::cend(seq_class_idxs),
                    [&num_classes](const int32_t t) { return t >= num_classes; })) {
        throw std::runtime_error{
                "seq_class_idxs=" + utils::print_container_as_string(seq_class_idxs, ",", true) +
                " containg out-of-range index for tensor with " + std::to_string(num_haplotypes) +
                " classes"};
    }

    const std::span<const float> data(probs_3D.data_ptr<float>(),
                                      static_cast<size_t>(probs_3D.numel()));
    if (std::empty(data)) {
        return 0.0f;
    }

    if ((blank_class_idx < 0) || (blank_class_idx >= num_classes)) {
        throw std::runtime_error("blank symbol has invalid class index.");
    }

    const int64_t seq_len = std::ssize(seq_class_idxs);
    if (seq_len > rlen) {
        return NEGINF;
    }

    const int64_t pos_stride = num_haplotypes * num_classes;
    const int64_t hap_offset = hap_id * num_classes;

    const auto log_prob = [&data](const int64_t offset, const int64_t class_id) {
        return std::log(std::max(EPS, data[offset + class_id]));
    };

    if (seq_len == 0) {
        float total = 0.0f;
        for (int64_t pos = rstart; pos < rend; ++pos) {
            const int64_t offset = pos * pos_stride + hap_offset;
            total += log_prob(offset, blank_class_idx);
        }
        return total;
    }

    std::vector<float> dp(static_cast<size_t>(seq_len + 1), NEGINF);
    dp[0] = 0.0f;

    for (int64_t rel_pos = 0; rel_pos < rlen; ++rel_pos) {
        const int64_t offset = (rstart + rel_pos) * pos_stride + hap_offset;
        const float blank_logprob = log_prob(offset, blank_class_idx);
        const int64_t min_emitted_after = std::max<int64_t>(0, seq_len - (rlen - rel_pos - 1));
        const int64_t max_emitted_after = std::min(seq_len, rel_pos + 1);

        for (int64_t emitted = max_emitted_after; emitted >= min_emitted_after; --emitted) {
            const float stay = dp[static_cast<size_t>(emitted)] + blank_logprob;
            float emit = NEGINF;
            if (emitted > 0) {
                emit = dp[static_cast<size_t>(emitted - 1)] +
                       log_prob(offset, seq_class_idxs[static_cast<size_t>(emitted - 1)]);
            }
            dp[static_cast<size_t>(emitted)] = logaddexp(stay, emit);
        }
        if (min_emitted_after > 0) {
            dp[static_cast<size_t>(min_emitted_after - 1)] = NEGINF;
        }
    }

    return dp[static_cast<size_t>(seq_len)];
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
    const int64_t num_classes = probs_3D.size(2);
    const int64_t blank_class_idx = symbol_lookup['*'];

    if (std::ssize(sequences) != num_haplotypes) {
        throw std::runtime_error(
                "Number of haplotypes in the tensor differs from the number of "
                "sequences provided to compute_quality_from_probs. Tensor shape: " +
                utils::tensor_shape_as_string(probs_3D) +
                ", number of sequences: " + std::to_string(std::size(sequences)));
    }

    float total = 0.0f;
    for (int64_t hap_id = 0; hap_id < num_haplotypes; ++hap_id) {
        const std::string_view subseq(sequences[hap_id].data() + rstart,
                                      static_cast<size_t>(rend - rstart));
        const std::vector<int32_t> seq_class_idxs = make_class_indices(
                subseq, symbol_lookup, num_classes, blank_class_idx, false, true);
        const float log_prob =
                compute_subseq_log_prob(probs_3D, seq_class_idxs, rstart, rend, hap_id);
        total += log_prob;
    }
    total = phred(-std::expm1(total), VCF_MAX_GQ_CAP);

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
 * \brief Compute the logprob matrix of all gap-compressed sequences in alleles
 *        for each haplotype in probs_3D
 * \param probs_3D A 3D tensor of probabilities (output of inference) for a single sample (not batch).
 *                  Dimensions: [seq_len x num_haplotypes x num_classes].
 * \param alleles Gap-compressed sequences.
 * \param symbol_lookup Lookup table of symbol chars (bases) -> numeric ID of that symbol, to encode sequence bases into class IDs.
 * \param rstart Region start (start of the subsequence in seq). Zero-based.
 * \param rend Region end (end of the subsequence in seq). Non-inclusive.
 * \returns Vector with linear representation of the 2-d array of log-probabilities [num_alleles, num_haplotypes].
 */
std::vector<float> compute_allele_logprobs_array(const at::Tensor& probs_3D,
                                                 const std::span<const std::string> alleles,
                                                 const std::array<int32_t, 256>& symbol_lookup,
                                                 const int64_t rstart,
                                                 const int64_t rend) {
    const int64_t num_positions = rend - rstart;
    const std::string padding(num_positions, '*');
    const int64_t num_haplotypes = probs_3D.size(1);
    const int64_t num_classes = probs_3D.size(2);
    const int64_t num_alleles = std::size(alleles);
    const int64_t blank = symbol_lookup['*'];

    std::vector<float> allele_logprobs(num_alleles * num_haplotypes, NEGINF);

    for (int64_t all_idx = 0; all_idx < std::ssize(alleles); ++all_idx) {
        std::string_view allele = alleles[all_idx];
        const int64_t offset = all_idx * num_haplotypes;
        const int64_t seq_len = std::ssize(allele);
        // Test whether there is only one unique decode path through the scores.
        // This will be the case when either all positions are gaps or there are no gaps.
        // If so, for speed we compute the log-probability using direct evaluation given
        // the class indices.
        // When there is a mixture of gaps and bases in the allele, there will be more
        // than one valid decode path. In this case, compute the log-probability using
        // dynamic programming as the sum over all valid paths.
        if ((seq_len == 0) || (seq_len == num_positions)) {
            if (seq_len == 0) {
                allele = std::string_view(padding.c_str(), num_positions);
            }
            DORADO_LOG(trace,
                       "[compute_allele_logprobs_array] using explicit path for seq={} in region "
                       "{}-{}",
                       allele, rstart, rend);
            const std::vector<int32_t> seq_class_idxs =
                    make_class_indices(allele, symbol_lookup, num_classes, blank, true, true);
            for (int64_t hap_idx = 0; hap_idx < num_haplotypes; ++hap_idx) {
                allele_logprobs[offset + hap_idx] =
                        compute_subseq_log_prob(probs_3D, seq_class_idxs, rstart, rend, hap_idx);
            }
        } else {
            DORADO_LOG(trace,
                       "[compute_allele_logprobs_array] using dp path for seq={} in region {}-{}",
                       allele, rstart, rend);
            const std::vector<int32_t> seq_class_idxs =
                    make_class_indices(allele, symbol_lookup, num_classes, blank, true, false);
            for (int64_t hap_idx = 0; hap_idx < num_haplotypes; ++hap_idx) {
                allele_logprobs[offset + hap_idx] = compute_subseq_log_prob_dp(
                        probs_3D, seq_class_idxs, blank, rstart, rend, hap_idx);
            }
        }
    }
    return allele_logprobs;
}

// Lookup table for ordering of genotype (Number="G") records as specified in VCF spec.
// See https://samtools.github.io/hts-specs/VCFv4.4.pdf
// Outer dimension of ALLELE_SETS is ploidy. Depending on the number of alleles
// we select a number of elements as specified in NUM_ELEMENTS. Each element
// shows the corresponding set of alleles contributing to the GL value at that
// position.
// Currently only enumerated up to num_alleles=3 (alts=2) for ploidy=1 and
// and num_alleles=4 (alts=3) for ploidy=2.
// clang-format off
const std::vector<std::vector<std::vector<int32_t>>> ALLELE_SETS = {
    /*ploidy=1*/ {{0}, {1}, {2}},
    /*ploidy=2*/ {
                     {0, 0},
                     {0, 1}, {1, 1},
                     {0, 2}, {1, 2}, {2, 2},
                     {0, 3}, {1, 3}, {2, 3}, {3, 3},
                 },
};
const std::vector<std::vector<int64_t>> NUM_ELEMENTS = {
    /*ploidy=1*/ {1, 2, 3},
    /*ploidy=2*/ {1, 3, 6, 10},
};
// clang-format on

/**
 * \brief Calculate total log-probability for all possible permutations of a single genotype.
 */
float compute_total_genotype_likelihood(const std::vector<float>& hap_allele_logprobs,
                                        const std::vector<int32_t>& genotype,
                                        const int64_t num_haplotypes) {
    if (std::empty(hap_allele_logprobs)) {
        throw std::runtime_error{"[compute_total_genotype_likelihood] Empty log probs"};
    }
    if (std::empty(genotype)) {
        return NEGINF;
    }
    const int64_t num_alleles = std::ssize(hap_allele_logprobs) / num_haplotypes;
    if (std::ssize(genotype) != num_haplotypes) {
        throw std::runtime_error{
                "[compute_total_genotype_likelihood] Number of genotype entries " +
                std::to_string(std::ssize(genotype)) +
                "doesn't match expected number of haplotypes: " + std::to_string(num_haplotypes)};
    }
    if (std::any_of(std::cbegin(genotype), std::cend(genotype), [&num_alleles](const auto all_id) {
            return (all_id < 0) || (all_id >= num_alleles);
        })) {
        throw std::runtime_error{"[compute_total_genotype_likelihood] Genotype " +
                                 utils::print_container_as_string(genotype, ",", true) +
                                 " contains entries out of range for " +
                                 std::to_string(num_alleles) + " alleles"};
    }
    // make local mutable copy
    std::vector<int32_t> gt = genotype;
    float total_logprobs = NEGINF;
    do {
        float perm_logprobs = 0.f;
        for (int64_t hap_id = 0; hap_id < std::ssize(gt); ++hap_id) {
            perm_logprobs += hap_allele_logprobs[gt[hap_id] * num_haplotypes + hap_id];
        }
        total_logprobs = logaddexp(total_logprobs, perm_logprobs);
    } while (std::next_permutation(std::begin(gt), std::end(gt)));
    return total_logprobs;
}

/**
 * \brief Compute an array of GL values from tensor of per-haplotype per-allele log-probabilities.
 */
std::vector<float> compute_GLs(const std::vector<float>& hap_allele_logprobs,
                               const int64_t num_haplotypes) {
    const int64_t num_alleles = std::ssize(hap_allele_logprobs) / num_haplotypes;
    const auto [num_elements, genotype_order] = lookup_gts(num_haplotypes, num_alleles);

    std::vector<float> genotype_likelihoods(num_elements, NEGINF);
    for (int64_t gt = 0; gt < num_elements; ++gt) {
        genotype_likelihoods[gt] = compute_total_genotype_likelihood(
                hap_allele_logprobs, genotype_order[gt], num_haplotypes);
    }
    return genotype_likelihoods;
}

/**
 * \brief Construct PL string from array of GL values.
 */
std::string construct_PL_string(const std::vector<float>& genotype_logprobs) {
    if (std::empty(genotype_logprobs)) {
        return {};
    }
    const int64_t num_genotypes = std::ssize(genotype_logprobs);
    const float max_gl =
            *std::max_element(std::cbegin(genotype_logprobs), std::cend(genotype_logprobs));
    std::vector<int32_t> pl_values;
    pl_values.reserve(num_genotypes);
    for (const float gl : genotype_logprobs) {
        pl_values.emplace_back(
                static_cast<int32_t>(std::round(phred_from_log(gl - max_gl, VCF_MAX_PL_CAP))));
    }
    return utils::print_container_as_string(pl_values, ",", false);
}

std::vector<float> calculate_AQs(const std::vector<float>& allele_logprobs,
                                 const int64_t num_haplotypes) {
    if (std::empty(allele_logprobs)) {
        return {};
    }
    const int64_t num_alleles = std::ssize(allele_logprobs) / num_haplotypes;
    std::vector<float> aq_values;
    aq_values.reserve(num_alleles);
    for (int32_t i = 0; i < num_alleles; ++i) {
        const int32_t offset = i * num_haplotypes;
        float logerr = 0.f;
        // calculate the joint log-probability that "i is wrong" over all alleles
        // clamp to avoid overflow for allele_logprobs very close to 0
        for (int32_t j = 0; j < num_haplotypes; ++j) {
            logerr += std::log(std::clamp(-std::expm1(allele_logprobs[offset + j]), EPS, 1.0f));
        }
        aq_values.emplace_back(phred_from_log(logerr, VCF_MAX_GQ_CAP));
    }
    return aq_values;
};

std::string construct_AQ_string(const std::vector<float>& aq_floats) {
    if (std::empty(aq_floats)) {
        return {};
    }
    std::vector<int32_t> aq_values;
    aq_values.reserve(std::size(aq_floats));
    for (const float aq : aq_floats) {
        aq_values.emplace_back(static_cast<int32_t>(std::round(aq)));
    }
    return utils::print_container_as_string(aq_values, ",", false);
};

/**
 * \brief QUAL is calculated from the likelihood that genotype is reference
 */
float get_qual(const std::vector<float>& genotype_logprobs, bool is_reference) {
    if (std::empty(genotype_logprobs)) {
        return 0.0f;
    }
    float err = 0.0f;
    if (is_reference) {
        err = -std::expm1(genotype_logprobs[0]);
    } else {
        err = std::exp(genotype_logprobs[0]);
    }
    return round_float(phred(err, VCF_MAX_GQ_CAP), 3);
}

/**
 * \brief GQ is calculated as the likelihood that the predicted genotype is wrong
 */
float calculate_GQ(const std::vector<float>& genotype_logprobs,
                   const std::vector<int32_t>& gt,
                   const int64_t num_alleles) {
    if (std::empty(genotype_logprobs) || std::empty(gt)) {
        return -1.0f;
    }

    const int32_t num_haplotypes = std::ssize(gt);
    const auto [num_elements, genotype_order] =
            dorado::secondary::lookup_gts(num_haplotypes, num_alleles);
    if ((num_elements == 0) || (num_elements > std::ssize(genotype_logprobs))) {
        return -1.0f;
    }
    const auto it_go = std::find(std::cbegin(genotype_order), std::cend(genotype_order), gt);
    if (it_go == std::cend(genotype_order)) {
        throw std::runtime_error{"Genotype " + utils::print_container_as_string(gt, "/", true) +
                                 " not found in genotype_order"};
    }
    const int64_t idx = std::distance(std::cbegin(genotype_order), it_go);
    // calculate log-probs excluding idx, then normalize including the predicted genotype
    float logerr = NEGINF;
    for (int64_t i = 0; i < num_elements; ++i) {
        if (i != idx) {
            logerr = logaddexp(logerr, genotype_logprobs[i]);
        }
    }
    logerr -= logaddexp(logerr, genotype_logprobs[idx]);
    return phred_from_log(logerr, VCF_MAX_GQ_CAP);
}

/**
 * \brief Get the minimum AQ value over the alleles in a specific genotype.
 */
float get_min_AQ(const std::vector<float>& aq_values, const std::vector<int32_t>& gt) {
    if (std::empty(aq_values)) {
        return -1.0f;
    }

    float min_aq = VCF_MAX_GQ_CAP;
    for (int32_t allele_idx : gt) {
        if ((allele_idx < 0) || (allele_idx >= std::ssize(aq_values))) {
            throw std::runtime_error{"Genotype " +
                                     utils::print_container_as_string(gt, "/", false) +
                                     " contains entries out of range for " +
                                     std::to_string(std::size(aq_values)) + " AQ values"};
        }
        if (aq_values[allele_idx] < min_aq) {
            min_aq = aq_values[allele_idx];
        }
    }
    return min_aq;
}

/**
 * \brief Get the phred-scaled total probability of a specific genotype.
 */
float calculate_VQ(const std::vector<float>& allele_logprobs, const std::vector<int32_t>& gt) {
    if (std::empty(allele_logprobs)) {
        return -1.0f;
    }

    float v_logprob = 0.0f;
    const int64_t num_haplotypes = std::ssize(gt);

    for (int32_t hap = 0; hap < num_haplotypes; ++hap) {
        const int64_t idx = gt[hap] * num_haplotypes + hap;
        if ((idx < 0) || (idx >= std::ssize(allele_logprobs))) {
            throw std::runtime_error{
                    "Genotype " + utils::print_container_as_string(gt, "/", false) +
                    " contains entries out of range for " +
                    std::to_string(std::size(allele_logprobs)) + " allele log-probabilities"};
        }
        v_logprob += allele_logprobs[idx];
    }
    return phred(-std::expm1(v_logprob), VCF_MAX_GQ_CAP);
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
                          const std::string& variant_source,
                          const at::Tensor& probs_3D,
                          const std::unordered_set<char>& symbol_set,
                          const std::array<int32_t, 256>& symbol_lookup,
                          const at::Tensor& basecalls,
                          const float pass_min_qual,
                          const float pass_min_gq,
                          const float pass_min_aq,
                          const float pass_min_vq) {
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

    if (rend == rstart) {
        return {};
    }

    const int64_t num_haplotypes = std::ssize(cons_seqs_with_gaps);

    // Get the reference and the predictions for the variant stretch.
    const std::string_view var_ref_with_gaps(std::data(ref_seq_with_gaps) + rstart,
                                             static_cast<size_t>(rend - rstart));

    // Mutable ref and pred sequences - a ref base may be prepended later.
    std::string var_ref = remove_gaps(var_ref_with_gaps);

    std::vector<std::string> var_preds;
    for (int64_t hap_id = 0; hap_id < num_haplotypes; ++hap_id) {
        const std::string_view var_pred_with_gaps(
                std::data(cons_seqs_with_gaps[hap_id].seq) + rstart,
                static_cast<size_t>(rend - rstart));
        var_preds.emplace_back(remove_gaps(var_pred_with_gaps));
    }

    const std::vector<std::string> alleles_lookup = construct_alleles_lookup(var_ref, var_preds);

    if (is_var && (std::size(alleles_lookup) == 1)) {
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

    const std::vector<std::string> alts_out = construct_alts_out(alleles_lookup);

    // Compute likelihood matrix for alleles across haplotypes
    const std::vector<float> hap_allele_likelihoods =
            compute_allele_logprobs_array(probs_3D, alleles_lookup, symbol_lookup, rstart, rend);
    const std::vector<float> genotype_likelihoods =
            compute_GLs(hap_allele_likelihoods, num_haplotypes);
    const std::string pl_str = construct_PL_string(genotype_likelihoods);
    const std::vector<float> aq_values = calculate_AQs(hap_allele_likelihoods, num_haplotypes);
    const std::string aq_str = construct_AQ_string(aq_values);
    const float qual = get_qual(genotype_likelihoods, false);

    // gt_ints preserves local haplotype phasing
    // VQ calculated before sorting gt_ints
    std::vector<int32_t> gt_ints = get_gt_indices(var_preds, alleles_lookup);
    const float min_aq = get_min_AQ(aq_values, gt_ints);
    const float vq_val = calculate_VQ(hap_allele_likelihoods, gt_ints);

    // sort gt_ints before calculating GQ so we can directly look up in ALLELE_SETS
    // and GT string is ordered
    std::sort(std::begin(gt_ints), std::end(gt_ints));
    const float gq_val = calculate_GQ(genotype_likelihoods, gt_ints, std::ssize(alleles_lookup));
    const std::string gt = utils::print_container_as_string(gt_ints, "/", false);

    Variant var{
            .seq_id = seq_id,
            .pos = positions_major[rstart],
            .ref = var_ref,
            .alts = alts_out,
            .filters = {qual < pass_min_qual ? Filters::LowQual : Filters::Pass},
            .info = {},
            .qual = qual,
            .format = {{"GT", gt}, {FORMAT_TAG_VARIANT_SOURCE, variant_source}},
            .rstart = rstart,
            .rend = rend,
    };

    // Check if variant starts on insert, prepend previous ref base.
    bool prepended = false;
    if (positions_minor[var.rstart] != 0) {
#ifdef DEBUG_NORMALIZE_VARIANT
        std::cerr << "[construct_variant] Running prepend_major_ref_base.\n";
#endif
        prepend_major_ref_base(var);
        prepended = true;
    }

    if (basecalls.defined()) {
        int64_t temp_rstart = var.rstart;
        int64_t temp_rend = var.rend;
        std::vector<std::string> temp_alleles;
        temp_alleles.reserve(std::size(var.alts) + 1);
        temp_alleles.emplace_back(var.ref);
        temp_alleles.insert(std::end(temp_alleles), std::begin(var.alts), std::end(var.alts));
        extend_span(temp_rstart, temp_rend, temp_alleles, prepended, draft, positions_major,
                    positions_minor, ref_seq_with_gaps, cons_seqs_with_gaps);

        const auto ad =
                calculate_basecall_allele_depths(temp_alleles, temp_rstart, temp_rend, basecalls);
        if (!std::empty(ad)) {
            var.format["AD"] = ad;
        }
    }

    if (normalize) {
        // Create a vector of views for normalize_variant.
        std::vector<std::string_view> cons_view;
        cons_view.reserve(num_haplotypes);
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

    if (gq_val >= 0) {
        var.format["GQ"] = q_string_from_float(gq_val);
        if (gq_val < pass_min_gq) {
            var.filters.update(Filters::LowGQ);
        }
    }
    if (!std::empty(pl_str)) {
        var.format["PL"] = pl_str;
    }
    if (!std::empty(aq_str)) {
        var.format["AQ"] = aq_str;
        if ((min_aq >= 0) && (min_aq < pass_min_aq)) {
            var.filters.update(Filters::LowAQ);
        }
    }
    if (vq_val >= 0) {
        var.format["VQ"] = fmt::format("{:.2f}", vq_val);
        if (vq_val < pass_min_vq) {
            var.filters.update(Filters::LowVQ);
        }
    }

    sanitize_empty_alts(var, num_haplotypes);

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
                                           const std::array<int32_t, 256>& symbol_lookup,
                                           const at::Tensor& basecalls,
                                           const float pass_min_qual,
                                           const float pass_min_gq,
                                           const float pass_min_aq,
                                           const float pass_min_vq) {
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
            furthest_rend = std::max<int64_t>(v2.rend, furthest_rend);
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

            const std::string variant_source = utils::get_value_or_default(
                    variants[prev_i].format, FORMAT_TAG_VARIANT_SOURCE, VARIANT_SOURCE_UNKNOWN);

            Variant new_var = construct_variant(
                    draft, positions_major, positions_minor, ref_seq_with_gaps, cons_seqs_with_gaps,
                    variants[prev_i].seq_id, v1.rstart, furthest_rend, true, ambig_ref, normalize,
                    variant_source, probs_3D, symbol_set, symbol_lookup, basecalls, pass_min_qual,
                    pass_min_gq, pass_min_aq, pass_min_vq);

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
            const std::string variant_source = utils::get_value_or_default(
                    variants[prev_i].format, FORMAT_TAG_VARIANT_SOURCE, VARIANT_SOURCE_UNKNOWN);

            Variant new_var = construct_variant(
                    draft, positions_major, positions_minor, ref_seq_with_gaps, cons_seqs_with_gaps,
                    variants[prev_i].seq_id, variants[prev_i].rstart, furthest_rend, true,
                    ambig_ref, normalize, variant_source, probs_3D, symbol_set, symbol_lookup,
                    basecalls, pass_min_qual, pass_min_gq, pass_min_aq, pass_min_vq);

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
    const int64_t prev_ref_pos = ref_pos - (positions_minor[rstart] > 0 ? 0 : 1);

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

void add_depth(Variant& var, const at::Tensor& depth) {
    // missing depth information
    if ((!depth.defined()) || (depth.size(0) == 0)) {
        return;
    }

    const auto depth_span = depth.index({at::indexing::Slice(var.rstart, var.rend)});
    // reference block, output MIN_DP
    if (is_reference_record(var)) {
        const int64_t dp = at::min(depth_span).item<int64_t>();
        var.format["MIN_DP"] = std::to_string(dp);
        return;
    }

    // default case, output DP and MIN_DP if multi-site and different
    const int64_t med_dp = at::median(depth_span).item<int64_t>();
    var.format["DP"] = std::to_string(med_dp);
    if ((var.rend - var.rstart) > 1) {
        const int64_t min_dp = at::min(depth_span).item<int64_t>();
        if (min_dp != med_dp) {
            var.format["MIN_DP"] = std::to_string(min_dp);
        }
    }
}

}  // namespace

/**
 * \brief Lookup genotypes list with bounds checking.
 */
std::pair<int64_t, std::span<const std::vector<int32_t>>> lookup_gts(const int64_t ploidy,
                                                                     const int64_t num_alleles) {
    if ((ploidy <= 0) || (ploidy > std::ssize(ALLELE_SETS))) {
        throw std::runtime_error{"ALLELE_SETS not implemented for ploidy " +
                                 std::to_string(ploidy)};
    }
    if ((ploidy > std::ssize(NUM_ELEMENTS)) || (num_alleles <= 0) ||
        (num_alleles > std::ssize(NUM_ELEMENTS[ploidy - 1]))) {
        throw std::runtime_error{"NUM_ELEMENTS not implemented for ploidy " +
                                 std::to_string(ploidy) + " and " + std::to_string(num_alleles) +
                                 " alleles"};
    }
    const int64_t num_elements = NUM_ELEMENTS[ploidy - 1][num_alleles - 1];
    const std::span<const std::vector<int32_t>> genotype_range{std::cbegin(ALLELE_SETS[ploidy - 1]),
                                                               static_cast<size_t>(num_elements)};
    return std::make_pair(num_elements, genotype_range);
}

std::string q_string_from_float(const float qual) {
    if (qual < 0.0f) {
        return ".";
    }
    return std::to_string(static_cast<int32_t>(std::round(qual)));
}

void extend_span(int64_t& rstart,
                 int64_t& rend,
                 std::vector<std::string>& alleles,
                 const bool prepended,
                 const std::string_view draft,
                 const std::vector<int64_t>& positions_major,
                 const std::vector<int64_t>& positions_minor,
                 const std::string_view ref_seq_with_gaps,
                 const std::vector<ConsensusResult>& cons_seqs_with_gaps) {
    if ((rstart < 0) || (rend <= rstart)) {
        throw std::runtime_error("[extend_span] Invalid (rstart, rend): (" +
                                 std::to_string(rstart) + "," + std::to_string(rend) + ")");
    }
    const int64_t ref_len = std::ssize(ref_seq_with_gaps);
    if (rend > ref_len) {
        throw std::runtime_error("[extend_span] rend " + std::to_string(rend) +
                                 " out of range for reference length " + std::to_string(ref_len));
    }
    if (std::any_of(std::cbegin(cons_seqs_with_gaps), std::cend(cons_seqs_with_gaps),
                    [&](const auto& cons_result) { return rend > std::ssize(cons_result.seq); })) {
        throw std::runtime_error("[extend_span] rend " + std::to_string(rend) +
                                 " out of range for consensus length");
    }

    const bool ref_has_front_gap = ref_seq_with_gaps[rstart] == '*';
    const bool ref_has_back_gap = ref_seq_with_gaps[rend - 1] == '*';
    const bool cons_has_front_gap = std::any_of(
            std::cbegin(cons_seqs_with_gaps), std::cend(cons_seqs_with_gaps),
            [&rstart](const auto& cons_result) { return cons_result.seq[rstart] == '*'; });
    const bool cons_has_back_gap = std::any_of(
            std::cbegin(cons_seqs_with_gaps), std::cend(cons_seqs_with_gaps),
            [&rend](const auto& cons_result) { return cons_result.seq[rend - 1] == '*'; });

    const bool left_indel = ref_has_front_gap || cons_has_front_gap;
    const bool right_indel = ref_has_back_gap || cons_has_back_gap;

    const int64_t num_positions = std::size(ref_seq_with_gaps);

    // Extend rstart and rend to include all minor positions immediately adjacent to the variant
    // ref                   A***CG***AC
    // alt                   A***GA***AC
    // initial rstart, rend      S E
    // final rstart, rend     S       E
    while ((!prepended) && (rstart > 0) && (positions_minor[rstart - 1] != 0)) {
        --rstart;
    }
    while ((rend < num_positions) && (positions_minor[rend] != 0)) {
        ++rend;
    }
    // If the variant is an indel, also include the flanking major base on either side, so we can
    // catch indels extending beyond the variant span
    // ref                   ACGTACGT
    // alt                   AC**ACGT
    // initial rstart, rend    S E
    // final rstart, rend     S   E
    if (left_indel && (!prepended) && (rstart > 0)) {
        --rstart;
        const char base = draft[positions_major[rstart]];
        for (auto& allele : alleles) {
            allele.insert(std::begin(allele), base);
        }
    }
    if (right_indel && (rend < num_positions)) {
        const char base = draft[positions_major[rend]];
        ++rend;
        for (auto& allele : alleles) {
            allele.push_back(base);
        }
    }
}

Variant make_gvcf_reference_record(const int32_t seq_id,
                                   const int64_t pos,
                                   const char ref_base,
                                   const float qual,
                                   const int32_t ploidy,
                                   const int32_t min_depth,
                                   const int64_t end,
                                   const int64_t rstart,
                                   const int64_t rend,
                                   const std::string_view variant_source) {
    const bool is_callable = utils::is_canonical_base(ref_base);
    const float qual_ = is_callable ? qual : -1.0f;

    const std::string gt_str = make_reference_gt(ploidy, is_callable ? '0' : '.');
    const std::string gq_str = q_string_from_float(qual_);
    const std::string depth_str = (min_depth >= 0) ? std::to_string(min_depth) : ".";
    Variant ref_var{
            .seq_id = seq_id,
            .pos = pos,
            .ref = std::string(1, ref_base),
            .alts = {"<*>"},
            .filters = {Filters::NonVariant},
            .info = {},
            .qual = qual_,
            .format = {{"GT", gt_str},
                       {"GQ", gq_str},
                       {FORMAT_TAG_VARIANT_SOURCE, std::string(variant_source)}},
            .rstart = rstart,
            .rend = rend,
    };
    set_gvcf_reference_block_end(ref_var, end);
    // this is done last to preserve the previous order of fields
    ref_var.format.emplace(std::make_pair("MIN_DP", depth_str));

    return ref_var;
}

void add_VAF(Variant& var) {
    const std::vector<float> vaf_values = compute_VAF_from_AD(var, VAF_DP);

    if (std::empty(vaf_values)) {
        return;
    }
    var.format["VAF"] = utils::print_container_as_string(vaf_values, ",", false);
}

void sanitize_empty_alts(Variant& var, const int32_t ploidy) {
    const bool any_empty = std::empty(var.alts) ||
                           std::any_of(std::cbegin(var.alts), std::cend(var.alts),
                                       [](const std::string_view s) { return std::empty(s); });
    if (any_empty) {
        DORADO_LOG(debug, "[sanitize_empty_alts] resetting variant: {}", to_string(var));
        var.alts = {"."};
        var.format["GT"] = make_reference_gt(ploidy, '0');
        var.format.erase("AD");
        var.format.erase("VAF");
        var.format.erase("PL");
        var.format.erase("AQ");
        var.format.erase("VQ");
        var.filters = {Filters::NonVariant};
    }
}

Variant collapse_to_haploid(const Variant& var, const bool require_hom) {
    Variant ret = var;

    // Find GT and GQ records for later
    auto it_gt = ret.format.find("GT");
    if (it_gt == ret.format.end()) {
        DORADO_LOG(warn, "Failed to find GT record for variant at ({}, {})", ret.seq_id, ret.pos);
        ret.alts.clear();
        return ret;
    }
    auto it_gq = ret.format.find("GQ");
    if (it_gq == ret.format.end()) {
        DORADO_LOG(warn, "Failed to find GQ record for variant at ({}, {})", ret.seq_id, ret.pos);
        ret.alts.clear();
        return ret;
    }

    // This is a gVCF record.
    if (is_reference_record(var)) {
        if (std::empty(ret.alts)) {
            ret.alts = {"."};
        }
        ret.filters = {Filters::NonVariant};
        const bool is_missing_call =
                (std::size(ret.ref) == 1) && !utils::is_canonical_base(ret.ref.front());
        if (is_missing_call) {
            ret.qual = -1.0f;
            it_gt->second = ".";
            it_gq->second = ".";
        } else {
            it_gt->second = "0";
        }
        return ret;
    }

    const int64_t num_called_alts =
            std::count_if(std::cbegin(var.alts), std::cend(var.alts),
                          [](const std::string& alt) { return alt != "<*>"; });
    if (num_called_alts > 1) {
        DORADO_LOG(debug, "Filtering variant with more than one alt in haploid region ({}, {}).",
                   var.seq_id, var.pos);
        ret.filters.update(Filters::IncompatZyg);
        ret.qual = 0.0f;
        it_gt->second = "0";
        it_gq->second = "0";
        return ret;
    }

    const bool is_homozygous = it_gt->second.find('0') == std::string::npos;
    if (is_homozygous || !require_hom) {
        it_gt->second = "1";
    } else {
        DORADO_LOG(debug, "Filtering heterozygous variant ({}, {}) in haploid region.", ret.seq_id,
                   ret.pos);
        ret.filters.update(Filters::IncompatZyg);
        ret.qual = 0.0f;
        it_gt->second = "0";
        it_gq->second = "0";
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
        DORADO_LOG(debug, "[normalize_variant] all_same_as_ref");
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
        var.pos += (rev ? 0 : start_pos);
    };

    const auto trim_end_and_align = [&ref_with_gaps, &cons_seqs_with_gaps, &positions_major,
                                     &positions_minor,
                                     &symbol_set](Variant& var, const bool ambig_ref_local) {
        const auto reset_var = [](const Variant& v) {
#ifdef DEBUG_NORMALIZE_VARIANT
            std::cerr << "[normalize_variant]    [trim_end_and_align] Resetting var.\n";
#endif
            std::vector<std::string> seqs{v.ref};
            seqs.insert(std::end(seqs), std::cbegin(v.alts), std::cend(v.alts));
            return std::make_pair(v, std::move(seqs));
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
                                           positions_minor, symbol_set, ambig_ref_local);

                if (!changed) {
                    // std::tie(var, seqs) = reset_var(var_before_change);
                    changed = append_ref_base(var, ref_with_gaps, cons_seqs_with_gaps,
                                              positions_major, positions_minor, symbol_set,
                                              ambig_ref_local);
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

    const auto normalize_variant_core = [&ref_with_gaps, &cons_seqs_with_gaps, &positions_major,
                                         &positions_minor, &trim_start, &trim_end_and_align](
                                                const Variant& var, const bool ambig_ref_local) {
        Variant ret = var;

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
            return Variant();
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
        std::cerr << "[normalize_variant] Entry variant: " << ret << '\n';
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
            trim_end_and_align(ret, ambig_ref_local);
        }

#ifdef DEBUG_NORMALIZE_VARIANT
        std::cerr << "[normalize_variant] Before trim_start 2:" << ret << '\n';
#endif

        trim_start(ret, false);

#ifdef DEBUG_NORMALIZE_VARIANT
        std::cerr << "[normalize_variant] Final variant: " << ret << '\n';
#endif

        return ret;
    };

    Variant ret = normalize_variant_core(variant, ambig_ref);

    // Check whether the ALT field is still empty. This can happen when a deletion is
    // flanked by unreachable positions on both ends. If so, force prepend a base and
    // retry normalisation without the ambig_ref constraint.
    const bool any_empty = std::any_of(std::cbegin(ret.alts), std::cend(ret.alts),
                                       [](const std::string_view val) { return std::empty(val); });
    if (any_empty && !ambig_ref) {
        const auto [can_go_left, new_rstart, new_ref_pos] =
                find_previous_ref_pos(positions_major, positions_minor, ret.rstart);

        if (can_go_left) {
            ret.pos = new_ref_pos;
            ret.rstart = new_rstart;
            const char base = ref_with_gaps[new_rstart];
            ret.ref = base + ret.ref;
            for (size_t i = 0; i < std::size(ret.alts); ++i) {
                ret.alts[i] = base + ret.alts[i];
            }
        }
        ret = normalize_variant_core(ret, true);
    }
    return ret;
}

std::vector<Variant> general_decode_variants(
        const DecoderBase& decoder,
        const VariantCallingSample& vc_sample,
        const std::string_view draft,
        const float pass_min_qual,
        const float pass_min_gq,
        const float pass_min_aq,
        const float pass_min_vq,
        const bool ambig_ref,
        const bool return_all,
        const bool normalize,
        const bool merge_overlapping,
        const bool merge_adjacent,
        const std::span<const std::pair<int64_t, float>> gvcf_reference_block_gq_margins) {
    // Validate inputs.
    vc_sample.validate();

    const int32_t seq_id = vc_sample.seq_id;
    const std::vector<int64_t>& positions_major = vc_sample.positions_major;
    const std::vector<int64_t>& positions_minor = vc_sample.positions_minor;
    const at::Tensor probs = vc_sample.logits;

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
        DORADO_LOG(
                warn,
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

    if constexpr (DEBUG_VC_SAMPLES == true) {
        constexpr bool PRINT_ALL = ((DEBUG_VC_SAMPLES_START < 0) || (DEBUG_VC_SAMPLES_END < 0));
        const bool print_this =
                PRINT_ALL || (((DEBUG_VC_SAMPLES_START >= 0) && (DEBUG_VC_SAMPLES_END > 0)) &&
                              (vc_sample.start() < DEBUG_VC_SAMPLES_END) &&
                              (DEBUG_VC_SAMPLES_START < vc_sample.end()));
        if (print_this) {
            std::vector<std::string_view> cons_view2;
            cons_view2.reserve(std::size(cons_seqs_with_gaps));
            for (const ConsensusResult& val : cons_seqs_with_gaps) {
                cons_view2.emplace_back(val.seq);
            }
            std::cerr << "[general_decode_variants] seq_id = " << seq_id
                      << ", columns = " << std::size(positions_major) << '\n';
            print_slice(std::cerr,
                        "Slice printed in general_decode_variants before constructing variants:",
                        ref_seq_with_gaps, cons_view2, positions_major, positions_minor, is_variant,
                        0, -1, 0, -1);
        }
    }

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
                                        ambig_ref, normalize, VARIANT_SOURCE_INFERENCE, probs_3D,
                                        symbol_set, symbol_lookup, vc_sample.basecalls,
                                        pass_min_qual, pass_min_gq, pass_min_aq, pass_min_vq);

#ifdef DEBUG_VARIANT_REGIONS
        {
            std::cerr << "[general_decode_variants] region: rstart = " << rstart
                      << ", rend = " << rend << ", is_var = " << is_var << '\n';
            std::cerr << "[general_decode_variants slice] var = " << var << '\n';
            const int64_t s = std::max<int64_t>(0, var.rstart - 5);
            const int64_t e = std::min(std::ssize(positions_major), var.rend + 5);
            print_slice(std::cerr,
                        "Slice printed in general_decode_variants after construct_variant:",
                        ref_seq_with_gaps, cons_view, positions_major, positions_minor, is_variant,
                        s, e, var.rstart, var.rend);
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
                                         symbol_set, symbol_lookup, vc_sample.basecalls,
                                         pass_min_qual, pass_min_gq, pass_min_aq, pass_min_vq);
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

            Variant var = make_gvcf_reference_record(
                    seq_id, block_start_pos, draft[block_start_pos], min_gq, num_haplotypes,
                    /*min_depth*/ -1, block_end_pos, block_start_idx, block_end_idx,
                    VARIANT_SOURCE_INFERENCE);
            variants.emplace_back(std::move(var));
        }
    }

    std::sort(std::begin(variants), std::end(variants));

    // Normalize variants.
    std::vector<Variant> normalized_variants;
    normalized_variants.reserve(std::size(variants));
    for (const Variant& var : variants) {
        Variant new_var = var;
        add_depth(new_var, vc_sample.depth);
        add_VAF(new_var);
        // Sanity check.
        if (is_valid(new_var)) {
            normalized_variants.emplace_back(std::move(new_var));
        }
    }

    return normalized_variants;
}

std::pair<std::vector<std::string_view>, bool> split_GT_string(const std::string_view gt) {
    std::vector<std::string_view> indices;
    if (std::empty(gt)) {
        return {indices, false};
    }
    if (gt.contains('|') && gt.contains('/')) {
        throw std::runtime_error{
                "Gentoype strings containing both `|` and `/` separators are unsupported."};
    }
    const char separator = gt.contains('|') ? '|' : '/';

    indices.reserve(std::size(gt));
    for (const std::string_view hap : utils::split_view(gt, separator)) {
        if (!std::empty(hap)) {
            indices.emplace_back(hap);
        }
    }
    return {indices, separator == '|'};
}

}  // namespace dorado::secondary
