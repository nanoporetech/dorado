#include "variant_calling.h"

#include "consensus_result.h"
#include "consensus_utils.h"
#include "sample_trimming.h"
#include "secondary/batching.h"
#include "utils/rle.h"
#include "utils/ssize.h"

#include <ATen/ATen.h>
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

namespace dorado::secondary {

namespace {

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

}  // namespace

Variant normalize_variant(const std::string_view ref_with_gaps,
                          const std::vector<std::string_view>& cons_seqs_with_gaps,
                          const std::vector<int64_t>& positions_major,
                          const std::vector<int64_t>& positions_minor,
                          const Variant& variant) {
    const bool all_same_as_ref =
            std::all_of(std::cbegin(variant.alts), std::cend(variant.alts),
                        [&variant](const std::string& s) { return s == variant.ref; });

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

        const int64_t min_len = dorado::ssize(
                *std::min_element(std::cbegin(seqs), std::cend(seqs),
                                  [](const std::string_view a, const std::string_view b) {
                                      return dorado::ssize(a) < dorado::ssize(b);
                                  }));

        // Never trim the last base.
        int64_t start_pos = 0;
        for (int64_t i = 0; i < (min_len - 1); ++i) {
            bool bases_same = true;
            for (size_t j = 1; j < std::size(seqs); ++j) {
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
        var.alts = std::vector<std::string>(std::begin(seqs) + 1, std::end(seqs));
        var.pos += start_pos;
    };

    const auto trim_end_and_align = [&ref_with_gaps, &cons_seqs_with_gaps, &positions_major,
                                     &positions_minor](Variant& var) {
        const auto find_previous_major = [&](int64_t rpos) {
            while (rpos > 0) {
                --rpos;
                if (positions_minor[rpos] == 0) {
                    break;
                }
            }
            return rpos;
        };

        const auto reset_var = [](const Variant& v) {
            std::vector<std::string> seqs{v.ref};
            seqs.insert(std::end(seqs), std::cbegin(v.alts), std::cend(v.alts));
            return std::make_pair(v, seqs);
        };

        std::vector<std::string> seqs{var.ref};
        seqs.insert(std::end(seqs), std::cbegin(var.alts), std::cend(var.alts));

        bool changed = true;

        while (changed) {
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
                for (size_t i = 1; i < std::size(seqs); ++i) {
                    if (seqs[i].back() != seqs[0].back()) {
                        all_same = false;
                        break;
                    }
                }

                // Trim right.
                if (all_same) {
                    for (auto& seq : seqs) {
                        seq.resize(dorado::ssize(seq) - 1);
                    }
                    changed = true;
                    var.ref = seqs[0];
                    var.alts = std::vector<std::string>(std::begin(seqs) + 1, std::end(seqs));
                }
            }

            const bool any_empty =
                    std::any_of(std::cbegin(seqs), std::cend(seqs),
                                [](const std::string_view s) { return std::empty(s); });

            // Extend. Prepend/append a base if any seq is empty.
            if (any_empty) {
                // If we can't extend to the left, take one reference base to the right.
                if ((var.pos == 0) || (var.rstart == 0)) {
                    // If this variant is at the beginning of the ref, append a ref base.
                    const int64_t ref_pos = dorado::ssize(seqs[0]);
                    int64_t found_idx = -1;
                    for (int64_t idx = 0; idx < dorado::ssize(positions_major); ++idx) {
                        if ((positions_major[idx] == ref_pos) && (positions_minor[idx] == 0)) {
                            found_idx = idx;
                            break;
                        }
                    }
                    changed = false;
                    if (found_idx >= 0) {
                        // Found a candidate base, append it.
                        const char base = ref_with_gaps[found_idx];
                        for (auto& seq : seqs) {
                            seq += base;
                        }
                        var.rend = found_idx + 1;
                        var.ref = seqs[0];
                        var.alts = std::vector<std::string>(std::begin(seqs) + 1, std::end(seqs));
                        changed = true;
                    } else {
                        // Revert any trimming and stop if a base wasn't found.
                        // E.g. the variant regions covers the full window.
                        std::tie(var, seqs) = reset_var(var_before_change);
                        changed = false;
                    }
                    break;
                } else {
                    const int64_t new_rstart = find_previous_major(var.rstart);
                    const int64_t span = var.rstart - new_rstart;

                    // Sanity check. Revert any trimming if for any reason we cannot extend to the left.
                    if (span == 0) {
                        std::tie(var, seqs) = reset_var(var_before_change);
                        changed = false;
                        break;
                    }

                    // Collect all prefixes for all consensus sequences and the ref.
                    std::vector<std::string> prefixes;
                    prefixes.emplace_back(ref_with_gaps.substr(new_rstart, span));
                    for (const auto& seq : cons_seqs_with_gaps) {
                        prefixes.emplace_back(seq.substr(new_rstart, span));
                    }

                    // Create a set to count unique prefixes.
                    const std::unordered_set<std::string> prefix_set(std::begin(prefixes),
                                                                     std::end(prefixes));

                    // Check if there is more than 1 unique prefix - this is a variant then, stop extension.
                    // Revert trimming if it was applied.
                    if (std::size(prefix_set) > 1) {
                        std::tie(var, seqs) = reset_var(var_before_change);
                        changed = false;
                        break;
                    }

                    // Remove the deletions and prepend the prefix sequence.
                    for (size_t i = 0; i < std::size(seqs); ++i) {
                        std::string& p = prefixes[i];
                        p.erase(std::remove(std::begin(p), std::end(p), '*'), std::end(p));
                        seqs[i] = p + seqs[i];
                    }

                    // Finally, extend to the left.
                    var.pos = positions_major[new_rstart];
                    var.rstart = new_rstart;
                    var.ref = seqs[0];
                    var.alts = std::vector<std::string>(std::begin(seqs) + 1, std::end(seqs));
                    changed = true;
                }
            }
        }
        var.ref = seqs[0];
        var.alts = std::vector<std::string>(std::begin(seqs) + 1, std::end(seqs));
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
    // While actually, the variant.pos should have been set to the first major position after rstart.
    if (!std::empty(ret.ref)) {
        for (int32_t r = ret.rstart; r < ret.rend; ++r) {
            if ((positions_major[r] >= ret.pos) && (positions_minor[r] == 0)) {
                ret.pos = positions_major[r];
                break;
            }
        }
    }

    if (std::empty(ref_with_gaps)) {
        trim_start(ret, true);
    } else {
        trim_end_and_align(ret);
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
    const std::string& cons_seq_with_gaps = c.front().seq;

    // Draft sequence with gaps.
    const std::string ref_seq_with_gaps =
            extract_draft_with_gaps(draft, vc_sample.positions_major, vc_sample.positions_minor);

    // Candidate variant positions.
    const std::vector<bool> is_variant =
            variant_columns(vc_sample.positions_minor, ref_seq_with_gaps, cons_seq_with_gaps);
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
        const std::string_view var_ref_with_gaps(std::data(ref_seq_with_gaps) + rstart,
                                                 static_cast<size_t>(rend - rstart));
        const std::string_view var_pred_with_gaps(std::data(cons_seq_with_gaps) + rstart,
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

        Variant var{
                vc_sample.seq_id,     var_pos,  var_ref, {var_pred}, "PASS", {},
                round_float(qual, 3), genotype, rstart,  rend,
        };

        // Variant starts on insert - prepend ref base.
        if ((vc_sample.positions_minor[var.rstart] != 0) && !std::empty(var.alts)) {
            while ((var.rstart > 0) && (vc_sample.positions_minor[var.rstart] != 0)) {
                --var.rstart;
            }
            var.pos = vc_sample.positions_major[var.rstart];
            var.ref = draft[var.pos] + var.ref;
            var.alts[0] = draft[var.pos] + var.alts[0];
        }

        var = normalize_variant(ref_seq_with_gaps, {cons_seq_with_gaps}, vc_sample.positions_major,
                                vc_sample.positions_minor, var);

        variants.emplace_back(std::move(var));
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
                {"."},
                ".",
                {},
                round_float(qual, 3),
                genotype,
                i, (i + 1),
            };
            // clang-format on

            variants.emplace_back(std::move(variant));
        }
    }

    std::sort(std::begin(variants), std::end(variants), [](const auto& a, const auto& b) {
        return std::tie(a.seq_id, a.pos) < std::tie(b.seq_id, b.pos);
    });

    return variants;
}

}  // namespace dorado::secondary