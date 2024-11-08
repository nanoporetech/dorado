#include "trim.h"

#include "utils/ssize.h"

#include <torch/torch.h>

#include <iostream>
#include <stdexcept>
#include <string>

namespace dorado::polisher {

enum class Relationship {
    DIFFERENT_REF_NAME,
    FORWARD_OVERLAP,
    REVERSE_OVERLAP,
    FORWARD_ABUTTED,
    REVERSE_ABUTTED,
    FORWARD_GAPPED,
    REVERSE_GAPPED,
    S2_WITHIN_S1,
    S1_WITHIN_S2,
    UNKNOWN,
};

Sample slice(const Sample& sample, const int64_t start_idx, std::optional<int64_t> end_idx) {
    Sample sliced_sample;

    // Handle features slicing
    if (sample.features.defined()) {
        sliced_sample.features =
                sample.features.slice(0, start_idx, end_idx.value_or(sample.features.size(0)));
    }

    // Handle positions_major slicing
    if (start_idx < dorado::ssize(sample.positions_major)) {
        auto end = end_idx.value_or(sample.positions_major.size());
        sliced_sample.positions_major.assign(sample.positions_major.begin() + start_idx,
                                             sample.positions_major.begin() + end);
    }

    // Handle positions_minor slicing
    if (start_idx < dorado::ssize(sample.positions_minor)) {
        auto end = end_idx.value_or(sample.positions_minor.size());
        sliced_sample.positions_minor.assign(sample.positions_minor.begin() + start_idx,
                                             sample.positions_minor.begin() + end);
    }

    // Handle depth slicing
    if (sample.depth.defined()) {
        sliced_sample.depth =
                sample.depth.slice(0, start_idx, end_idx.value_or(sample.depth.size(0)));
    }

    sliced_sample.seq_id = sample.seq_id;  // Copy the seq_id without modification

    return sliced_sample;
}

Relationship relative_position(const Sample& s1, const Sample& s2) {
    // Helper lambdas for comparisons
    const auto ordered_abuts = [](const Sample& s1_, const Sample& s2_) -> bool {
        const auto [s1_end_maj, s1_end_min] = s1_.get_last_position();
        const auto [s2_start_maj, s2_start_min] = s2_.get_position(0);
        if (((s2_start_maj == (s1_end_maj + 1)) && (s2_start_min == 0)) ||
            ((s2_start_maj == s1_end_maj) && (s2_start_min == (s1_end_min + 1)))) {
            return true;
        }
        return false;
    };

    const auto ordered_contained = [](const Sample& s1_, const Sample& s2_) -> bool {
        return (s2_.get_position(0) >= s1_.get_position(0)) &&
               (s2_.get_last_position() <= s1_.get_last_position());
    };

    const auto ordered_overlaps = [](const Sample& s1_, const Sample& s2_) -> bool {
        const auto [s1_end_maj, s1_end_min] = s1_.get_last_position();
        const auto [s2_start_maj, s2_start_min] = s2_.get_position(0);
        if ((s2_start_maj < s1_end_maj) ||
            ((s2_start_maj == s1_end_maj) && (s2_start_min < (s1_end_min + 1)))) {
            return true;
        }
        return false;
    };

    const auto ordered_gapped = [](const Sample& s1_, const Sample& s2_) -> bool {
        const auto [s1_end_maj, s1_end_min] = s1_.get_last_position();
        const auto [s2_start_maj, s2_start_min] = s2_.get_position(0);
        if ((s2_start_maj > (s1_end_maj + 1)) ||
            ((s2_start_maj > s1_end_maj) && (s2_start_min > 0)) ||
            ((s2_start_maj == s1_end_maj) && (s2_start_min > (s1_end_min + 1)))) {
            return true;
        }
        return false;
    };

    // Main logic for relative position determination
    if (s1.seq_id != s2.seq_id) {
        return Relationship::DIFFERENT_REF_NAME;
    }

    // Sort s1 and s2 by first position, and then by size in descending order
    const Sample& s1_ord = (std::tuple(s1.get_position(0), -dorado::ssize(s1.positions_major)) <=
                            std::tuple(s2.get_position(0), -dorado::ssize(s2.positions_major)))
                                   ? s1
                                   : s2;
    const Sample& s2_ord = (&s1_ord == &s1) ? s2 : s1;
    const bool is_ordered = (&s1_ord == &s1);

    // Determine the relationship based on various conditions
    if (ordered_contained(s1_ord, s2_ord)) {
        return is_ordered ? Relationship::S2_WITHIN_S1 : Relationship::S1_WITHIN_S2;
    } else if (ordered_abuts(s1_ord, s2_ord)) {
        return is_ordered ? Relationship::FORWARD_ABUTTED : Relationship::REVERSE_ABUTTED;
    } else if (ordered_overlaps(s1_ord, s2_ord)) {
        return is_ordered ? Relationship::FORWARD_OVERLAP : Relationship::REVERSE_OVERLAP;
    } else if (ordered_gapped(s1_ord, s2_ord)) {
        return is_ordered ? Relationship::FORWARD_GAPPED : Relationship::REVERSE_GAPPED;
    }
    return Relationship::UNKNOWN;
}

std::tuple<int64_t, int64_t, bool> overlap_indices(const Sample& s1, const Sample& s2) {
    const Relationship rel = relative_position(s1, s2);

    if (rel == Relationship::FORWARD_ABUTTED) {
        return {-1, -1, false};
    }

    if (rel != Relationship::FORWARD_OVERLAP) {
        throw std::runtime_error("Cannot overlap samples! Relationship is not FORWARD_OVERLAP.");
    }

    // Finding where the overlap starts and ends using std::lower_bound
    const auto it_ovl_start_ind1 =
            std::lower_bound(std::begin(s1.positions_major), std::end(s1.positions_major),
                             s2.positions_major.front());
    const auto it_ovl_end_ind2 =
            std::lower_bound(std::begin(s2.positions_major), std::end(s2.positions_major),
                             s1.positions_major.back());

    if ((it_ovl_start_ind1 == std::end(s1.positions_major)) ||
        (it_ovl_end_ind2 == std::end(s2.positions_major))) {
        throw std::runtime_error(
                "Samples should be overlappnig, but cannot find adequate cooordinate positions!");
    }

    const int64_t ovl_start_ind1 = std::distance(std::begin(s1.positions_major), it_ovl_start_ind1);
    const int64_t ovl_end_ind2 = std::distance(std::begin(s2.positions_major), it_ovl_end_ind2);

    int64_t end_1_ind = -1;
    int64_t start_2_ind = -1;

    const auto compare_subvectors = [](const std::vector<int64_t>& a, const int64_t a_start,
                                       const int64_t a_end, const std::vector<int64_t>& b,
                                       const int64_t b_start, const int64_t b_end) {
        if ((a_end - a_start) != (b_end - b_start)) {
            return false;
        }
        for (int64_t i = a_start, j = b_start; i < a_end; ++i, ++j) {
            if (a[i] != b[j]) {
                return false;
            }
        }
        return true;
    };

    bool heuristic = false;

    if (!compare_subvectors(s1.positions_minor, ovl_start_ind1, std::size(s1.positions_minor),
                            s2.positions_minor, 0, ovl_end_ind2)) {
        // In this case, overlaps are not equal in structure.
        heuristic = true;
    }

    if (!heuristic) {
        // Take midpoint as break point.
        const int64_t overlap_len = ovl_end_ind2;  // Both are equal in size.
        const int64_t pad_1 = overlap_len / 2;
        const int64_t pad_2 = overlap_len - pad_1;
        end_1_ind = ovl_start_ind1 + pad_1;
        start_2_ind = ovl_end_ind2 - pad_2;

        if (((end_1_ind - ovl_start_ind1) + (ovl_end_ind2 - start_2_ind)) != overlap_len) {
            end_1_ind = -1;
            start_2_ind = -1;
            heuristic = true;
        }
    }

    if (heuristic) {
        // Some sample producing methods will not create 1-to-1 mappings
        // in their sets of columns, e.g. where chunking has affected the
        // reads used. Here we find a split point near the middle where
        // the two halfs have the same number of minor positions
        // (i.e. look similar).
        // Require seeing a number of major positions.
        // Heuristic: find midpoint with similar major position counts.

        constexpr int32_t UNIQ_MAJ = 3;

        end_1_ind = -1;
        start_2_ind = -1;

        const auto count_unique = [](const std::vector<int64_t>& a, const int64_t start,
                                     const int64_t end) -> int64_t {
            const int64_t len = dorado::ssize(a);
            if (std::empty(a) || (end <= start) || (start >= len) || (end > len)) {
                return 0;
            }
            int64_t prev = a[start];
            int64_t ret = 1;
            for (int64_t i = (start + 1); i < end; ++i) {
                if (a[i] == prev) {
                    continue;
                }
                prev = a[i];
                ++ret;
            }
            return ret;
        };
        const auto streak_count = [](const std::vector<int64_t>& a,
                                     const int64_t start) -> int64_t {
            const int64_t len = dorado::ssize(a);
            if (std::empty(a) || (start >= len)) {
                return 0;
            }
            int64_t ret = 1;
            for (int64_t i = (start + 1); i < len; ++i) {
                if (a[i] != a[start]) {
                    break;
                }
            }
            return ret;
        };

        const int64_t unique_s1 =
                count_unique(s1.positions_major, ovl_start_ind1, std::size(s1.positions_minor));
        const int64_t unique_s2 = count_unique(s2.positions_minor, 0, ovl_end_ind2);

        if ((unique_s1 > UNIQ_MAJ) && (unique_s2 > UNIQ_MAJ)) {
            const int64_t start = s1.positions_major[ovl_start_ind1];
            const int64_t end = s1.positions_major.back();
            const int64_t mid = start + (end - start) / 2;
            int64_t offset = 1;

            while (end_1_ind == -1) {
                if (((mid + offset) > end) && ((mid - offset) < start)) {
                    break;
                }

                for (const int64_t test : {+offset, -offset}) {
                    const auto left = std::lower_bound(std::begin(s1.positions_major),
                                                       std::end(s1.positions_major), mid + test);
                    const auto right = std::lower_bound(std::begin(s2.positions_major),
                                                        std::end(s2.positions_major), mid + test);

                    const int64_t left_pos = std::distance(std::begin(s1.positions_major), left);
                    const int64_t right_pos = std::distance(std::begin(s2.positions_major), right);

                    const int64_t left_streak = streak_count(s1.positions_major, left_pos);
                    const int64_t right_streak = streak_count(s2.positions_major, right_pos);

                    if ((left != std::end(s1.positions_major)) &&
                        (right != std::end(s2.positions_major)) && (left_streak == right_streak)) {
                        end_1_ind = left_pos;
                        start_2_ind = right_pos;
                        break;
                    }
                }

                offset += 1;
            }
        }
    }

    // If return coordinates are -1, then a viable junction was not found.
    return {end_1_ind, start_2_ind, heuristic};
}

std::vector<std::tuple<Sample, bool, bool>> trim_samples(const std::vector<Sample>& samples) {
    std::vector<std::tuple<Sample, bool, bool>> result;

    if (std::empty(samples)) {
        return result;
    }

    size_t idx_s1 = 0;
    int64_t start_1 = -1;
    int64_t start_2 = -1;

    for (size_t i = 1; i < std::size(samples); ++i) {
        const Sample& s1 = samples[idx_s1];
        const Sample& s2 = samples[i];
        bool is_last_in_contig = false;
        int64_t end_1 = -1;
        bool heuristic = false;

        const Relationship rel = relative_position(s1, s2);

        if (rel == Relationship::S2_WITHIN_S1) {
            continue;

        } else if (rel == Relationship::FORWARD_OVERLAP) {
            std::tie(end_1, start_2, heuristic) = overlap_indices(s1, s2);

        } else if (rel == Relationship::FORWARD_GAPPED) {
            is_last_in_contig = true;
            end_1 = -1;
            start_2 = -1;
        } else {
            try {
                std::tie(end_1, start_2, heuristic) = overlap_indices(s1, s2);
            } catch (const std::runtime_error& e) {
                throw std::runtime_error("Unhandled overlap type whilst stitching chunks.");
            }
        }

        result.emplace_back(slice(s1, start_1, end_1), is_last_in_contig, heuristic);

        idx_s1 = i;
        start_1 = start_2;
    }

    {
        const Sample& s1 = samples[idx_s1];
        const bool is_last_in_contig = true;
        const int64_t end_1 = -1;
        const bool heuristic = false;
        result.emplace_back(slice(s1, start_1, end_1), is_last_in_contig, heuristic);
    }

    return result;
}

}  // namespace dorado::polisher
