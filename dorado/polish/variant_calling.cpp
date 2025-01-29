#include "variant_calling.h"

#include "consensus_result.h"
#include "polish_stats.h"
#include "trim.h"
#include "utils/rle.h"
#include "utils/ssize.h"

#include <cxxpool.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace dorado::polisher {

/**
 * \brief Copy the draft sequence for a given sample, and expand it with '*' in places of gaps.
 */
std::string extract_draft_with_gaps(const std::string& draft,
                                    const std::vector<int64_t>& positions_major,
                                    const std::vector<int64_t>& positions_minor) {
    if (std::size(positions_major) != std::size(positions_minor)) {
        throw std::runtime_error(
                "The positions_major and positions_minor are not of the same size! "
                "positions_major.size = " +
                std::to_string(std::size(positions_major)) +
                ", positions_minor.size = " + std::to_string(std::size(positions_minor)));
    }

    std::string ret(std::size(positions_major), '*');

    for (int64_t i = 0; i < dorado::ssize(positions_major); ++i) {
        ret[i] = (positions_minor[i] == 0) ? draft[positions_major[i]] : '*';
    }

    return ret;
}

std::string extract_draft(const std::string& draft, const std::vector<int64_t>& positions_major) {
    std::string ret(std::size(positions_major), '*');
    for (int64_t i = 0; i < dorado::ssize(positions_major); ++i) {
        ret[i] = draft[positions_major[i]];
    }
    return ret;
}

VariantCallingSample slice_vc_sample(const VariantCallingSample& vc_sample,
                                     const int64_t idx_start,
                                     const int64_t idx_end) {
    // Check that all members of the sample are of the same length.
    vc_sample.validate();

    const int64_t num_columns = dorado::ssize(vc_sample.positions_major);

    // Validate idx.
    if ((idx_start < 0) || (idx_start >= num_columns) || (idx_start >= idx_end) ||
        (idx_end > num_columns)) {
        throw std::out_of_range("Index is out of range in slice_vc_sample. idx_start = " +
                                std::to_string(idx_start) +
                                ", idx_end = " + std::to_string(idx_end) +
                                ", num_columns = " + std::to_string(num_columns));
    }

    // Slice.
    return VariantCallingSample{
            vc_sample.seq_id,
            std::vector<int64_t>(std::begin(vc_sample.positions_major) + idx_start,
                                 std::begin(vc_sample.positions_major) + idx_end),
            std::vector<int64_t>(std::begin(vc_sample.positions_minor) + idx_start,
                                 std::begin(vc_sample.positions_minor) + idx_end),
            vc_sample.logits.index({at::indexing::Slice(idx_start, idx_end)}).clone()};
}

std::vector<VariantCallingSample> merge_vc_samples(
        const std::vector<VariantCallingSample>& vc_samples) {
    const auto merge_adjacent_samples_in_place = [](VariantCallingSample& lh,
                                                    const VariantCallingSample& rh) {
        if (lh.seq_id != rh.seq_id) {
            std::ostringstream oss;
            oss << "Cannot merge samples. Different seq_id. lh = " << lh << ", rh = " << rh;
            throw std::runtime_error(oss.str());
        }
        if (lh.end() != (rh.start() + 1)) {
            std::ostringstream oss;
            oss << "Cannot merge samples, coordinates are not adjacent. lh = " << lh
                << ", rh = " << rh;
            throw std::runtime_error(oss.str());
        }

        const size_t width = std::size(lh.positions_major);

        // Insert positions vectors.
        lh.positions_major.reserve(width + std::size(rh.positions_major));
        lh.positions_major.insert(std::end(lh.positions_major), std::begin(rh.positions_major),
                                  std::end(rh.positions_major));
        lh.positions_minor.reserve(width + std::size(rh.positions_minor));
        lh.positions_minor.insert(std::end(lh.positions_minor), std::begin(rh.positions_minor),
                                  std::end(rh.positions_minor));

        // Merge the tensors.
        lh.logits = torch::cat({std::move(lh.logits), rh.logits});
    };

    if (std::empty(vc_samples)) {
        return {};
    }

    std::vector<VariantCallingSample> ret{vc_samples.front()};

    for (int64_t i = 1; i < dorado::ssize(vc_samples); ++i) {
        if (ret.back().end() == (vc_samples[i].start() + 1)) {
            merge_adjacent_samples_in_place(ret.back(), vc_samples[i]);
        } else {
            ret.emplace_back(vc_samples[i]);
        }
    }

    return ret;
}

/**
 * \brief This function restructures the neighboring samples for one draft sequence.
 *          Each input sample is split on the last non-variant position.
 *          The left part is merged with anything previously added to a queue (i.e.
 *          previous sample's right portion). The right part is then added to a cleared queue.
 *          The goal of this function is to prevent calling variants on sample boundaries.
 */
std::vector<VariantCallingSample> join_samples(const std::vector<VariantCallingSample>& vc_samples,
                                               const std::string& draft,
                                               const DecoderBase& decoder) {
    std::vector<VariantCallingSample> ret;

    std::vector<VariantCallingSample> queue;

    for (int64_t i = 0; i < dorado::ssize(vc_samples); ++i) {
        const VariantCallingSample& vc_sample = vc_samples[i];

        vc_sample.validate();

        // Unsqueeze the logits because this vector contains logits for each individual sample of the shape
        // [positions x class_probabilities], whereas the decode_bases function expects that the first dimension is
        // the batch sample ID. That is, the tensor should be of shape: [batch_sample_id x positions x class_probabilities].
        // In this case, the "batch size" is 1.
        const at::Tensor logits = vc_sample.logits.unsqueeze(0);
        const std::vector<ConsensusResult> c = decoder.decode_bases(logits);

        // This shouldn't be possible.
        if (std::size(c) != 1) {
            spdlog::warn(
                    "Unexpected number of consensus sequences generated from a single sample: "
                    "c.size = {}. Skipping consensus of this sample.",
                    std::size(c));
            continue;
        }

        // Sequences for comparison.
        const std::string& call_with_gaps = c.front().seq;
        const std::string draft_with_gaps = extract_draft_with_gaps(
                draft, vc_sample.positions_major, vc_sample.positions_minor);
        assert(std::size(call_with_gaps) == std::size(draft_with_gaps));

        const auto check_is_diff = [](const char base1, const char base2) {
            return (base1 != base2) || (base1 == '*' && base2 == '*');
        };

        // Check if all positions are diffs, or if all positions are gaps in both sequences.
        {
            int64_t count = 0;
            for (int64_t j = 0; j < dorado::ssize(call_with_gaps); ++j) {
                if (check_is_diff(call_with_gaps[j], draft_with_gaps[j])) {
                    ++count;
                }
            }
            if (count == dorado::ssize(call_with_gaps)) {
                // Merge the entire sample with the next one. We need at least one non-diff non-gap pos.
                queue.emplace_back(vc_sample);
                continue;
            }
        }

        const int64_t num_positions = dorado::ssize(vc_sample.positions_major);

        // Find a location where to split the sample.
        int64_t last_non_var_start = 0;
        for (int64_t j = (num_positions - 1); j >= 0; --j) {
            if ((vc_sample.positions_minor[j] == 0) &&
                !check_is_diff(call_with_gaps[j], draft_with_gaps[j])) {
                last_non_var_start = j;
                break;
            }
        }

        // Split the sample.
        VariantCallingSample left_slice = slice_vc_sample(vc_sample, 0, last_non_var_start);
        VariantCallingSample right_slice =
                slice_vc_sample(vc_sample, last_non_var_start, num_positions);

        // Enqueue the queue if possible.
        if (last_non_var_start > 0) {
            queue.emplace_back(std::move(left_slice));
        }

        // Merge and insert.
        if (!std::empty(queue)) {
            auto new_samples = merge_vc_samples(queue);
            queue.clear();

            ret.insert(std::end(ret), std::make_move_iterator(std::begin(new_samples)),
                       std::make_move_iterator(std::end(new_samples)));
        }

        // Reset the queue.
        queue = {std::move(right_slice)};
    }

    // Merge and insert.
    if (!std::empty(queue)) {
        auto new_samples = merge_vc_samples(queue);
        queue.clear();

        ret.insert(std::end(ret), std::make_move_iterator(std::begin(new_samples)),
                   std::make_move_iterator(std::end(new_samples)));
    }

    return ret;
}

std::vector<bool> variant_columns(const std::vector<int64_t>& minor,
                                  const std::string& reference,
                                  const std::string& prediction) {
    const bool lengths_valid = (std::size(minor) == std::size(reference)) &&
                               (std::size(reference) == std::size(prediction));
    if (!lengths_valid) {
        std::ostringstream oss;
        oss << "Cannot find variant columns because sequences are not of equal length. minor.size "
               "= "
            << std::size(minor) << ", reference.size = " << std::size(reference)
            << ", prediction.size = " << std::size(prediction);
        throw std::runtime_error(oss.str());
    }

    const int64_t len = dorado::ssize(prediction);
    std::vector<bool> ret(len, false);

    int64_t insert_length = 0;
    bool is_var = (reference[0] != prediction[0]);  // Assume start on major.
    ret[0] = is_var;

    for (int64_t i = 1; i < len; ++i) {
        if (minor[i] == 0) {
            // Start of new reference position.
            if (is_var) {
                // If we saw any vars in an insert run, set all inserts to true.
                for (int64_t j = (i - insert_length); j < i; ++j) {
                    ret[j] = true;
                }
            }
            is_var = (reference[i] != prediction[i]);
            ret[i] = is_var;
            insert_length = 0;
        } else {
            insert_length += 1;
            is_var = (is_var || (reference[i] != prediction[i]));
        }
    }

    // Set any remaining inserts.
    if (is_var) {
        for (int64_t j = (len - insert_length); j <= (len - 1); ++j) {
            ret[j] = true;
        }
    }

    return ret;
}

Variant normalize_variant(const Variant& variant, const std::string& ref_seq) {
    if (variant.alt == variant.ref) {
        return variant;
    }

    const auto trim_start = [](Variant& var, const bool rev) {
        std::array<std::string, 2> seqs{var.ref, var.alt};
        if (rev) {
            std::reverse(std::begin(seqs[0]), std::end(seqs[0]));
            std::reverse(std::begin(seqs[1]), std::end(seqs[1]));
        }
        const int64_t min_len = std::min(dorado::ssize(seqs[0]), dorado::ssize(seqs[1]));
        int64_t start_pos = 0;
        for (int64_t i = 0; i < (min_len - 1); ++i) {
            if (seqs[0][i] != seqs[1][i]) {
                break;
            }
            ++start_pos;
        }
        var.ref = seqs[0].substr(start_pos);
        var.alt = seqs[1].substr(start_pos);
        if (rev) {
            std::reverse(std::begin(var.ref), std::end(var.ref));
            std::reverse(std::begin(var.alt), std::end(var.alt));
        } else {
            var.pos += start_pos;
        }
    };

    const auto trim_end_and_align = [](Variant& var, const std::string& reference) {
        std::array<std::string, 2> seqs{var.ref, var.alt};
        bool changed = true;
        while (changed) {
            changed = false;

            // Trim the last base if identical.
            if (!std::empty(seqs[0]) && !std::empty(seqs[1]) &&
                (seqs[0].back() == seqs[1].back())) {
                seqs[0] = seqs[0].substr(0, dorado::ssize(seqs[0]) - 1);
                seqs[1] = seqs[1].substr(0, dorado::ssize(seqs[1]) - 1);
                changed = true;
            }

            if (std::empty(seqs[0]) || std::empty(seqs[1])) {
                if (var.pos == 0) {
                    seqs[0] += reference[std::size(seqs[0])];
                    seqs[1] += reference[std::size(seqs[1])];
                    break;
                } else {
                    --var.pos;
                    seqs[0] = reference[var.pos] + seqs[0];
                    seqs[1] = reference[var.pos] + seqs[1];
                    changed = true;
                }
            }
        }
        var.ref = seqs[0];
        var.alt = seqs[1];
    };

    Variant ret = variant;

    if (std::empty(ref_seq)) {
        trim_start(ret, true);
    } else {
        trim_end_and_align(ret, ref_seq);
    }

    trim_start(ret, false);

    return ret;
}

std::vector<Variant> decode_variants(const DecoderBase& decoder,
                                     const VariantCallingSample& vc_sample,
                                     const std::string& draft,
                                     const bool ambig_ref,
                                     const bool gvcf) {
    // Validate that all vectors/tensors are of equal length.
    vc_sample.validate();

    // No work to do.
    if (std::empty(vc_sample.positions_major)) {
        return {};
    }

    // Check that the sample begins on a non-insertion base.
    if (vc_sample.positions_minor.front() != 0) {
        std::ostringstream oss;
        oss << "The first position of a sample must not be an insertion. sample = " << vc_sample;
        throw std::runtime_error(oss.str());
    }

    // Helper lambdas.
    const auto remove_gaps = [](const std::string_view seq) {
        std::string ret;
        ret.reserve(std::size(seq));
        std::copy_if(std::begin(seq), std::end(seq), std::back_inserter(ret),
                     [](char c) { return c != '*'; });
        return ret;
    };
    const auto is_subset_of_symbols = [](const std::unordered_set<char>& symbol_map,
                                         const std::string& query) {
        for (const char c : query) {
            if (symbol_map.count(c) == 0) {
                return false;
            }
        }
        return true;
    };
    const auto create_symbol_lookup = [](const std::string& symbols) {
        std::array<int32_t, 256> ret;
        std::fill(std::begin(ret), std::end(ret), -1);
        for (int32_t i = 0; i < static_cast<int32_t>(std::size(symbols)); ++i) {
            ret[static_cast<int32_t>(symbols[i])] = i;
        }
        return ret;
    };
    const auto encode_seq = [](const std::array<int32_t, 256>& symbol_lookup,
                               const std::string_view seq, const bool substitute_n) {
        std::vector<int32_t> ret(std::size(seq));
        for (int64_t i = 0; i < dorado::ssize(seq); ++i) {
            const char c = (substitute_n && (seq[i] == 'N')) ? '*' : seq[i];
            ret[i] = symbol_lookup[c];
        }
        return ret;
    };
    const auto phred = [](float err, const float cap) {
        err = std::clamp(err, std::pow(10.0f, -cap / 10.0f), 1.0f);
        const float q = -10.0f * std::log10(err);
        return std::min(q, cap);
    };
    const auto compute_seq_quality = [&encode_seq, &phred](
                                             const std::array<int32_t, 256>& symbol_lookup,
                                             const at::Tensor& class_probs,
                                             const std::string_view seq, const bool substitute_n) {
        const std::vector<int32_t> encoded = encode_seq(symbol_lookup, seq, substitute_n);
        float sum = 0.0f;
        for (size_t i = 0; i < std::size(encoded); ++i) {
            const int32_t j = encoded[i];
            const float prob = class_probs[i][j].item<float>();
            const float err = 1.0f - prob;
            sum += phred(err, 70.0f);
        }
        return sum;
    };
    const auto round_float = [](float val, const int32_t decimal_places) {
        const float f = std::pow(10.0f, static_cast<float>(decimal_places));
        val = std::round(val * f) / f;
        return val;
    };

    // Alphabet for the label scheme.
    const std::string symbols = decoder.get_label_scheme_symbols();
    const std::unordered_set<char> symbol_set(std::begin(symbols), std::end(symbols));
    const std::array<int32_t, 256> symbol_lookup = create_symbol_lookup(symbols);

    // Predicted sequence with gaps.
    const at::Tensor logits = vc_sample.logits.unsqueeze(0);
    const std::vector<ConsensusResult> c = decoder.decode_bases(logits);
    const std::string& prediction = c.front().seq;

    // Draft sequence with gaps.
    const std::string reference =
            extract_draft_with_gaps(draft, vc_sample.positions_major, vc_sample.positions_minor);

    // Candidate variant positions.
    const std::vector<bool> is_variant =
            variant_columns(vc_sample.positions_minor, reference, prediction);
    const std::vector<std::tuple<int64_t, int64_t, bool>> runs =
            dorado::run_length_encode(is_variant);

    // Extract variants.
    std::vector<Variant> variants;
    for (const auto& [rstart, rend, is_var] : runs) {
        // Skip non-variants.
        if (!is_var) {
            continue;
        }

        // Get the reference and the predictions for the variant stretch.
        const std::string_view var_ref_with_gaps(std::data(reference) + rstart,
                                                 static_cast<size_t>(rend - rstart));
        const std::string_view var_pred_with_gaps(std::data(prediction) + rstart,
                                                  static_cast<size_t>(rend - rstart));

        // Mutable ref and pred sequences - a ref base may be prepended later.
        std::string var_ref = remove_gaps(var_ref_with_gaps);
        std::string var_pred = remove_gaps(var_pred_with_gaps);

        // Verbatim comment from Medaka:
        //      "del followed by insertion can lead to non-variant"
        //      "maybe skip things if reference contains ambiguous symbols"
        if ((var_ref == var_pred) && is_var) {
            continue;
        } else if (!ambig_ref && !is_subset_of_symbols(symbol_set, var_ref)) {
            continue;
        }

        // Calculate probabilities.
        const at::Tensor var_probs = vc_sample.logits.slice(0, rstart, rend);
        const float ref_qv = compute_seq_quality(symbol_lookup, var_probs, var_ref_with_gaps, true);
        const float pred_qv =
                compute_seq_quality(symbol_lookup, var_probs, var_pred_with_gaps, false);

        // Variant data.
        const float qual = pred_qv - ref_qv;
        const int32_t qual_i = static_cast<int32_t>(std::round(qual));
        const std::vector<std::pair<std::string, int32_t>> genotype{
                {"GT", 1},
                {"GQ", qual_i},
        };
        const int64_t var_pos = vc_sample.positions_major[rstart];
        if (vc_sample.positions_minor[rstart] != 0) {
            // Variant starts on insert - prepend ref base.
            var_ref.insert(0, 1, draft[var_pos]);
            var_pred.insert(0, 1, draft[var_pos]);
        }
        Variant variant{
                vc_sample.seq_id,     var_pos,  var_ref, var_pred, "PASS", {},
                round_float(qual, 3), genotype,
        };
        variant = normalize_variant(variant, draft);
        variants.emplace_back(std::move(variant));
    }

    if (gvcf) {
        for (int64_t i = 0; i < dorado::ssize(vc_sample.positions_major); ++i) {
            // Skip non-reference positions.
            if (vc_sample.positions_minor[i] != 0) {
                continue;
            }

            const int64_t pos = vc_sample.positions_major[i];
            const std::string ref(1, draft[pos]);
            const int32_t ref_encoded =
                    (draft[pos] != 'N') ? symbol_lookup[draft[pos]] : symbol_lookup['*'];

            const float qual = phred(1.0f - vc_sample.logits[i][ref_encoded].item<float>(), 70.0f);
            const int32_t qual_i = static_cast<int32_t>(std::round(qual));

            const std::vector<std::pair<std::string, int32_t>> genotype{
                    {"GT", 0},
                    {"GQ", qual_i},
            };

            // clang-format off
            Variant variant{
                vc_sample.seq_id,
                pos,
                ref,
                ".",
                ".",
                {},
                round_float(qual, 3),
                genotype,
            };
            // clang-format on

            variants.emplace_back(std::move(variant));
        }
    }

    return variants;
}

std::vector<VariantCallingSample> trim_vc_samples(
        const std::vector<VariantCallingSample>& vc_input_data,
        const std::vector<std::pair<int64_t, int32_t>>& group) {
    // Mock the Sample objects. Trimming works on Sample objects only, but
    // it only needs positions, not the actual tensors.
    std::vector<Sample> local_samples;
    local_samples.reserve(std::size(group));
    for (const auto& [start, id] : group) {
        const auto& vc_sample = vc_input_data[id];
        local_samples.emplace_back(Sample(vc_sample.seq_id, {}, vc_sample.positions_major,
                                          vc_sample.positions_minor, {}, {}, {}));
    }

    // Compute trimming of all samples for this group.
    const std::vector<TrimInfo> trims = trim_samples(local_samples, std::nullopt);

    std::vector<VariantCallingSample> trimmed_samples;

    assert(std::size(trims) == std::size(local_samples));
    assert(std::size(trims) == std::size(group));

    for (int64_t i = 0; i < dorado::ssize(trims); ++i) {
        const int32_t id = group[i].second;
        const auto& s = vc_input_data[id];
        const TrimInfo& t = trims[i];

        // Make sure that all vectors and tensors are of the same length.
        s.validate();

        const int64_t num_columns = dorado::ssize(s.positions_major);
        if ((t.start < 0) || (t.start >= num_columns) || (t.start >= t.end) ||
            (t.end > num_columns)) {
            throw std::out_of_range("Index is out of range in trim_vc_samples. idx_start = " +
                                    std::to_string(t.start) +
                                    ", idx_end = " + std::to_string(t.end) +
                                    ", num_columns = " + std::to_string(num_columns));
        }

        trimmed_samples.emplace_back(VariantCallingSample{
                s.seq_id,
                std::vector<int64_t>(std::begin(s.positions_major) + t.start,
                                     std::begin(s.positions_major) + t.end),
                std::vector<int64_t>(std::begin(s.positions_minor) + t.start,
                                     std::begin(s.positions_minor) + t.end),
                s.logits.index({at::indexing::Slice(t.start, t.end)}).clone()});
    }

    return trimmed_samples;
}

std::vector<Variant> call_variants(
        const dorado::polisher::Interval& region_batch,
        const std::vector<VariantCallingSample>& vc_input_data,
        const std::vector<std::unique_ptr<hts_io::FastxRandomReader>>& draft_readers,
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        const DecoderBase& decoder,
        const bool ambig_ref,
        const bool gvcf,
        const int32_t num_threads,
        PolishStats& polish_stats) {
    // Group samples by sequence ID.
    std::vector<std::vector<std::pair<int64_t, int32_t>>> groups(region_batch.length());
    for (int32_t i = 0; i < dorado::ssize(vc_input_data); ++i) {
        const auto& vc_sample = vc_input_data[i];

        const int32_t local_id = vc_sample.seq_id - region_batch.start;

        // Skip filtered samples.
        if (vc_sample.seq_id < 0) {
            continue;
        }

        if ((vc_sample.seq_id >= dorado::ssize(draft_lens)) || (local_id < 0) ||
            (local_id >= dorado::ssize(groups))) {
            spdlog::error(
                    "Draft ID out of bounds! r.draft_id = {}, draft_lens.size = {}, "
                    "groups.size = {}",
                    vc_sample.seq_id, std::size(draft_lens), std::size(groups));
            continue;
        }
        groups[local_id].emplace_back(vc_sample.start(), i);
    }

    // Worker for parallel processing.
    const auto worker = [&](const int32_t tid, const int32_t start, const int32_t end,
                            std::vector<std::vector<Variant>>& results, PolishStats& ps) {
        if ((start < 0) || (start >= end) || (end > dorado::ssize(results))) {
            throw std::runtime_error("Worker group_id is out of bounds! start = " +
                                     std::to_string(start) + ", end = " + std::to_string(end) +
                                     ", results.size = " + std::to_string(std::size(results)));
        }

        for (int32_t group_id = start; group_id < end; ++group_id) {
            const int64_t seq_id = group_id + region_batch.start;
            const std::string& header = draft_lens[seq_id].first;

            // Sort the group by start positions.
            auto& group = groups[group_id];
            std::stable_sort(std::begin(group), std::end(group));

            if (std::empty(group)) {
                continue;
            }

            // Get the draft sequence.
            const std::string draft = draft_readers[tid]->fetch_seq(header);

            // Trim the overlapping portions between samples.
            const auto trimmed_vc_samples = trim_vc_samples(vc_input_data, group);

            // Break and merge samples on non-variant positions.
            // const auto joined_samples = join_samples(vc_input_data, group, trims, draft, decoder);
            const auto joined_samples = join_samples(trimmed_vc_samples, draft, decoder);

            for (const auto& vc_sample : joined_samples) {
                std::vector<Variant> variants =
                        decode_variants(decoder, vc_sample, draft, ambig_ref, gvcf);

                ps.add("processed", static_cast<double>(vc_sample.end() - vc_sample.start()));

                results[group_id].insert(std::end(results[group_id]),
                                         std::make_move_iterator(std::begin(variants)),
                                         std::make_move_iterator(std::end(variants)));
            }
        }
    };

    // Partition groups to chunks for multithreaded processing.
    const std::vector<Interval> thread_chunks =
            compute_partitions(static_cast<int32_t>(std::size(groups)), num_threads);

    // Create the thread pool.
    cxxpool::thread_pool pool{std::size(thread_chunks)};

    // Create the futures.
    std::vector<std::future<void>> futures;
    futures.reserve(std::size(thread_chunks));

    // Reserve the space for results for each individual group.
    std::vector<std::vector<Variant>> thread_results(std::size(groups));

    // Add worker tasks.
    for (int32_t tid = 0; tid < static_cast<int32_t>(std::size(thread_chunks)); ++tid) {
        const auto [chunk_start, chunk_end] = thread_chunks[tid];
        futures.emplace_back(pool.push(worker, tid, chunk_start, chunk_end,
                                       std::ref(thread_results), std::ref(polish_stats)));
    }

    // Join and catch exceptions.
    try {
        for (auto& f : futures) {
            f.get();
        }
    } catch (const std::exception& e) {
        throw std::runtime_error{std::string("Caught exception when computing variants: ") +
                                 e.what()};
    }

    // Flatten the results.
    std::vector<Variant> all_results;
    {
        size_t count = 0;
        for (const auto& vals : thread_results) {
            count += std::size(vals);
        }
        all_results.reserve(count);
        for (auto& vals : thread_results) {
            all_results.insert(std::end(all_results), std::make_move_iterator(std::begin(vals)),
                               std::make_move_iterator(std::end(vals)));
        }
    }

    return all_results;
}

}  // namespace dorado::polisher