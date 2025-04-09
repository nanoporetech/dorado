#include "consensus_utils.h"

#include "utils/ssize.h"

#include <cassert>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace dorado::secondary {

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

    const int64_t draft_len = dorado::ssize(draft);

    std::string ret(std::size(positions_major), '*');

    for (int64_t i = 0; i < dorado::ssize(positions_major); ++i) {
        if ((positions_major[i] < 0) || (positions_major[i] >= draft_len)) {
            throw std::runtime_error(
                    "The positions_major contains coordinates out of range for the input draft! "
                    "Requested coordinate: " +
                    std::to_string(positions_major[i]) +
                    ", draft len = " + std::to_string(draft_len));
        }
        ret[i] = (positions_minor[i] == 0) ? draft[positions_major[i]] : '*';
    }

    return ret;
}

std::vector<bool> variant_columns(const std::vector<int64_t>& minor,
                                  const std::string_view reference,
                                  const std::string_view prediction) {
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

    if (std::empty(prediction)) {
        return {};
    }

    const int64_t len = dorado::ssize(prediction);
    std::vector<bool> ret(len, false);

    bool is_var = (reference[0] != prediction[0]);  // Assume start on major.
    ret[0] = is_var;
    int64_t insert_length = 0;

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
            ++insert_length;
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

std::vector<bool> find_polyploid_variants(
        const std::vector<int64_t>& positions_minor,
        const std::string_view ref_seq_with_gaps,
        const std::vector<std::string_view>& cons_seqs_with_gaps,
        const std::optional<std::unordered_set<char>>& allowed_symbol_set) {
    if (std::empty(cons_seqs_with_gaps)) {
        return {};
    }

    assert(std::size(positions_minor) == std::size(ref_seq_with_gaps));
    assert(std::size(ref_seq_with_gaps) == std::size(cons_seqs_with_gaps.front()));

    std::vector<bool> ret(std::size(positions_minor));

    for (size_t i = 0; i < std::size(cons_seqs_with_gaps); ++i) {
        const std::vector<bool> is_var =
                variant_columns(positions_minor, ref_seq_with_gaps, cons_seqs_with_gaps[i]);
        for (size_t j = 0; j < std::size(is_var); ++j) {
            const bool valid =
                    !allowed_symbol_set || (allowed_symbol_set->count(ref_seq_with_gaps[j]) > 0);
            ret[j] = (ret[j] || is_var[j]) && valid;
        }
    }

    return ret;
}

std::vector<bool> find_polyploid_variants(
        const std::vector<int64_t>& positions_minor,
        const std::string_view ref_seq_with_gaps,
        const std::vector<ConsensusResult>& cons_seqs_with_gaps,
        const std::optional<std::unordered_set<char>>& allowed_symbol_set) {
    // Convert the objects to views for the find_polyploid_variants interface.
    std::vector<std::string_view> cons_views;
    cons_views.reserve(std::size(cons_seqs_with_gaps));
    for (const auto& val : cons_seqs_with_gaps) {
        cons_views.emplace_back(val.seq);
    }

    return find_polyploid_variants(positions_minor, ref_seq_with_gaps, cons_views,
                                   allowed_symbol_set);
}

void print_slice(std::ostream& os,
                 const std::string_view ref_seq_with_gaps,
                 const std::vector<std::string_view>& cons_seqs,
                 const std::vector<int64_t>& pos_major,
                 const std::vector<int64_t>& pos_minor,
                 const std::vector<bool>& is_var,
                 const int64_t slice_start,
                 int64_t slice_end,
                 const int64_t rstart,
                 int64_t rend) {
    // Helper lambda to format a number with spacing if >= 10
    const auto format_val = [](const int64_t val) -> std::string {
        if (val < 10) {
            return std::to_string(val);
        }
        std::ostringstream oss;
        oss << ' ' << val;
        return oss.str();
    };

    const auto print_subvector = [&os, &format_val](const std::string_view prefix, const auto& vec,
                                                    const int64_t start, const int64_t end,
                                                    const bool add_spaces) {
        os << prefix;
        for (int64_t i = start; i < end; ++i) {
            if (add_spaces) {
                os << format_val(vec[i]);
            } else {
                os << vec[i];
            }
        }
        os << '\n';
    };

    slice_end = (slice_end <= 0) ? dorado::ssize(pos_major) : slice_end;
    rend = (rend <= 0) ? dorado::ssize(pos_major) : rend;

    // IDX
    os << "    - IDX  : ";
    for (int64_t i = slice_start; i < slice_start; ++i) {
        os << format_val(i);
    }
    os << '\n';

    print_subvector("    - MAJOR: ", pos_major, slice_start, slice_end, true);
    print_subvector("    - MINOR: ", pos_minor, slice_start, slice_end, true);
    print_subvector("    - REF:   ", ref_seq_with_gaps, slice_start, slice_end, false);
    for (size_t i = 0; i < cons_seqs.size(); ++i) {
        print_subvector("    - HAP " + std::to_string(i) + ": ", cons_seqs[i], slice_start,
                        slice_end, false);
    }
    print_subvector("    - VAR:   ", is_var, slice_start, slice_end, true);

    if ((rstart >= 0) && (rend > 0)) {
        os << "             " << std::string(rstart - slice_start, ' ')
           << std::string(rend - rstart, '^') << '\n';
    }
}

}  // namespace dorado::secondary
