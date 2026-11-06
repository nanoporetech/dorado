#include "secondary/consensus/sample.h"

#include "secondary/common/interval.h"
#include "torch_utils/tensor_utils.h"
#include "utils/log_utils.h"

#include <ATen/TensorIndexing.h>

#include <ostream>
#include <span>
#include <stdexcept>

namespace dorado::secondary {

void Sample::validate() const {
    const int64_t num_columns = std::ssize(positions_major);

    if (!features.defined()) {
        throw std::runtime_error("Sample::features tensor is not defined!");
    }

    if (!depth.defined()) {
        throw std::runtime_error("Sample::depth tensor is not defined!");
    }

    // Validate that the input data is sane.
    if ((std::ssize(positions_minor) != num_columns) || (depth.size(0) != num_columns) ||
        (features.size(0) != num_columns)) {
        throw std::runtime_error(
                "Sample data dimensions are inconsistent. positions_major.size = " +
                std::to_string(std::size(positions_major)) +
                ", positions_minor.size = " + std::to_string(std::size(positions_minor)) +
                ", depth.size(0) = " + std::to_string(depth.size(0)) +
                ", features.size(0) = " + std::to_string(features.size(0)));
    }
}

int64_t Sample::find_max_depth(const int64_t start_idx, const int64_t end_idx) const {
    if (!depth.defined()) {
        throw std::runtime_error{
                "Cannot compute max depth because Sample::depth tensor is not defined!"};
    }
    if ((start_idx < 0) || (end_idx <= start_idx) || (end_idx > depth.size(0))) {
        throw std::runtime_error{
                "Cannot compute max depth because start_idx/end_idx out of range. start_idx = " +
                std::to_string(start_idx) + ", end_idx = " + std::to_string(end_idx) +
                ", depth.shape = " + utils::tensor_shape_as_string(depth)};
    }
    const std::span<int64_t> depth_vals(depth.data_ptr<int64_t>(),
                                        static_cast<size_t>(depth.size(0)));
    int64_t ret = depth_vals[start_idx];
    for (int64_t i = start_idx; i < end_idx; ++i) {
        ret = std::max(ret, depth_vals[i]);
    }
    return ret;
}

at::Tensor Sample::select_feature_channel(const FeatureColumns& feature_column,
                                          const FeatureColumnMap& feature_column_map) const {
    if ((!features.defined()) || (features.dim() != 3) || std::empty(feature_column_map)) {
        return {};
    }
    const int64_t base_column =
            secondary::get_feature_column_or_throw(feature_column_map, feature_column);
    if ((base_column < 0) || (base_column >= features.size(-1))) {
        return {};
    }
    // contiguous called so that direct linear access works without striding
    return features.select(/*dim=*/-1, base_column).contiguous();
}

// Helper function to generate placeholder read IDs for read level models.
std::vector<std::string> placeholder_read_ids(const int64_t n) {
    std::vector<std::string> placeholder_ids(n);
    for (int64_t i = 0; i < n; ++i) {
        placeholder_ids[i] = "__placeholder_" + std::to_string(i);
    }
    return placeholder_ids;
}

Sample slice_sample(const Sample& sample,
                    const int64_t idx_start,
                    const int64_t idx_end,
                    const bool clone) {
    sample.validate();

    // Validate idx.
    const int64_t num_columns = std::ssize(sample.positions_major);
    if ((idx_start < 0) || (idx_start >= num_columns) || (idx_start >= idx_end) ||
        (idx_end > num_columns)) {
        throw std::out_of_range(
                "Index is out of range in slice_sample. idx_start = " + std::to_string(idx_start) +
                ", idx_end = " + std::to_string(idx_end) +
                ", num_columns = " + std::to_string(num_columns));
    }

    std::vector<std::string> placeholder_rids =
            ((idx_start > 0) || (idx_end < num_columns))
                    ? placeholder_read_ids(std::ssize(sample.read_ids_left))
                    : std::vector<std::string>{};

    // Create the sliced Sample.
    Sample ret{
            .seq_id = sample.seq_id,
            .features = sample.features.index({at::indexing::Slice(idx_start, idx_end)}),
            .positions_major = std::vector<int64_t>(std::begin(sample.positions_major) + idx_start,
                                                    std::begin(sample.positions_major) + idx_end),
            .positions_minor = std::vector<int64_t>(std::begin(sample.positions_minor) + idx_start,
                                                    std::begin(sample.positions_minor) + idx_end),
            .depth = sample.depth.index({at::indexing::Slice(idx_start, idx_end)}),
            .read_ids_left = (idx_start == 0) ? sample.read_ids_left : placeholder_rids,
            .read_ids_right = (idx_end == num_columns) ? sample.read_ids_right : placeholder_rids,
    };

    if (clone) {
        ret.features = ret.features.clone();
        ret.depth = ret.depth.clone();
    }

    if (sample.draft_seq) {
        ret.draft_seq = {sample.draft_seq->index({at::indexing::Slice(idx_start, idx_end)})};
        if (clone) {
            ret.draft_seq = {ret.draft_seq->clone()};
        }
    }

    return ret;
}

Sample slice_sample(const Sample& sample, const int64_t idx_start, const int64_t idx_end) {
    return slice_sample(sample, idx_start, idx_end, true);
}

void debug_print_sample(std::ostream& os,
                        const Sample& sample,
                        int64_t start /*= 0*/,
                        int64_t end /*= -1 */,
                        bool debug /*= false */) {
    const int64_t len = static_cast<int64_t>(std::size(sample.positions_major));
    start = std::max<int64_t>(0, start);
    end = (end <= 0) ? len : end;

    os << "sample.positions = " << sample.start() << " - " << sample.end()
       << " , dist = " << (sample.end() - sample.start()) << " , tensor = [";
    os.flush();
    for (int64_t k = start; k < std::min<int64_t>(start + 3, len); ++k) {
        os << "(" << sample.positions_major[k] << ", " << sample.positions_minor[k] << ") ";
        os.flush();
    }
    os << " ...";
    os.flush();
    for (int64_t k = std::max<int64_t>(0, end - 3); k < end; ++k) {
        os << " (" << sample.positions_major[k] << ", " << sample.positions_minor[k] << ")";
        os.flush();
    }
    os << "], size = " << std::size(sample.positions_major);
    os << ", depth.shape = " << utils::tensor_shape_as_string(sample.depth);
    os.flush();

    if (debug) {
        const auto depth = sample.depth.slice(/*dim=*/0, /*start=*/0);
        for (int64_t k = 0; k < len; ++k) {
            os << "[k = " << k << "] pos = (" << sample.positions_major[k] << ", "
               << sample.positions_minor[k] << "), depth = " << depth[k].item<float>() << '\n';
            os.flush();
        }
    }
}

std::ostream& operator<<(std::ostream& os, const Sample& sample) {
    debug_print_sample(os, sample, 0, -1, false);
    return os;
}

std::string sample_to_string(const Sample& sample) {
    std::ostringstream oss;
    oss << sample;
    return oss.str();
}

void merge_adjacent_samples_in_place(Sample& lh, const Sample& rh) {
    if (lh.seq_id != rh.seq_id) {
        std::ostringstream oss;
        oss << "Cannot merge samples. Different seq_id. lh = " << lh << ", rh = " << rh;
        throw std::runtime_error(oss.str());
    }
    if (lh.end() != (rh.start() + 1)) {
        std::ostringstream oss;
        oss << "Cannot merge samples, coordinates are not adjacent. lh = " << lh << ", rh = " << rh;
        throw std::runtime_error(oss.str());
    }

    const size_t width = std::size(lh.positions_major);

    // Merge the tensors.
    lh.features = at::cat({std::move(lh.features), rh.features});
    lh.depth = at::cat({std::move(lh.depth), rh.depth});

    if ((lh.draft_seq) && (rh.draft_seq)) {
        lh.draft_seq = {at::cat({std::move(*lh.draft_seq), *rh.draft_seq})};
    } else {
        lh.draft_seq = {std::nullopt};
    }

    // Insert positions vectors.
    lh.positions_major.reserve(width + std::size(rh.positions_major));
    lh.positions_major.insert(std::end(lh.positions_major), std::begin(rh.positions_major),
                              std::end(rh.positions_major));
    lh.positions_minor.reserve(width + std::size(rh.positions_minor));
    lh.positions_minor.insert(std::end(lh.positions_minor), std::begin(rh.positions_minor),
                              std::end(rh.positions_minor));

    // Propagate read IDs.
    lh.read_ids_right = rh.read_ids_right;
}

std::vector<secondary::Sample> split_samples(std::vector<Sample> samples,
                                             const int64_t chunk_len,
                                             const int64_t chunk_overlap) {
    if (chunk_len <= 0) {
        throw std::runtime_error("Wrong chunk_len. chunk_len = " + std::to_string(chunk_len));
    }
    if ((chunk_overlap < 0) || (chunk_overlap > chunk_len)) {
        throw std::runtime_error(
                "Wrong chunk_overlap length. chunk_len = " + std::to_string(chunk_len) +
                ", chunk_overlap = " + std::to_string(chunk_overlap));
    }

    std::vector<secondary::Sample> results;
    results.reserve(std::size(samples));

    for (auto& sample : samples) {
        const int64_t sample_len = static_cast<int64_t>(std::size(sample.positions_major));

        if (sample_len <= chunk_len) {
            results.emplace_back(std::move(sample));
            continue;
        }

        const int64_t step = chunk_len - chunk_overlap;

        // Slice out all but the last chunk unless perfectly sized.
        int64_t end = 0;
        for (int64_t start = 0; start < (sample_len - chunk_len + 1); start += step) {
            end = start + chunk_len;
            results.emplace_back(slice_sample(sample, start, end, false));
        }

        // Last chunk will have a large overlap with previous, to maintain equal length.
        if (end < sample_len) {
            const int64_t start = sample_len - chunk_len;
            end = sample_len;
            results.emplace_back(slice_sample(sample, start, end, false));
        }
    }

    return results;
}

std::vector<Sample> split_samples_around_positions(
        std::vector<Sample> samples,
        const secondary::IntervalTreesInt64Map& candidate_trees,
        const int64_t chunk_len,
        const int64_t flanking_bases) {
    constexpr int64_t MIN_FLANKING_BASES = 3;

    if (chunk_len <= 0) {
        throw std::runtime_error("Wrong chunk_len. chunk_len = " + std::to_string(chunk_len));
    }
    if ((flanking_bases < 0) || (flanking_bases > chunk_len)) {
        throw std::runtime_error(
                "Wrong flanking_bases length. chunk_len = " + std::to_string(chunk_len) +
                ", flanking_bases = " + std::to_string(flanking_bases));
    }

    if (std::empty(candidate_trees)) {
        return {};
    }

    const auto searchsorted_left = [](const std::vector<int64_t>& vec, const int64_t x) -> int64_t {
        const auto it = std::lower_bound(std::cbegin(vec), std::cend(vec), x);
        return static_cast<std::int64_t>(std::distance(std::cbegin(vec), it));
    };

    std::vector<secondary::Sample> all_results;
    all_results.reserve(std::size(samples));

    // Retained chunks own cloned tensors, so release each source after it has been processed.
    for (auto& sample : samples) {
        const auto it_seq_id = candidate_trees.find(sample.seq_id);

        // No candidate positions for this sequence.
        if (it_seq_id == std::cend(candidate_trees)) {
            sample = {};
            continue;
        }
        const auto& tree = it_seq_id->second;

        // Get all candidate positions for this region.
        // Note: this interval tree lib uses inclusive end coordinate.
        std::vector<interval_tree::Interval<int64_t, int64_t>> positions =
                tree.findOverlapping(sample.start(), sample.end() - 1);

        // Sort the positions in ascending order.
        std::sort(std::begin(positions), std::end(positions),
                  [](const auto& a, const auto& b) { return a.start < b.start; });

        std::vector<secondary::Sample> results;
        std::vector<int64_t> last_flanking_bases;

#ifdef DEBUG_POLISH_SPLIT_SAMPLES_AROUND_POSITIONS
        DORADO_LOG(debug, "[split_samples_around_positions] Input sample: {}",
                   secondary::sample_to_string(sample));
        for (int64_t i = 0; i < std::ssize(positions); ++i) {
            DORADO_LOG(debug,
                       "[split_samples_around_positions]     [candidate i = {}] position = {}", i,
                       positions[i].start);
        }
#endif

        bool stop_chunking = false;
        for (const auto itvl : positions) {
            // Walk through the entire candidate span (e.g. a long deletion), creating chunks until
            // all of its positions within this sample are covered.
            const int64_t span_end = std::min(itvl.stop, sample.end() - 1);
            int64_t position = std::max(itvl.start, sample.start());
            while (position <= span_end) {
                // Skip positions which are already covered by the previous chunk.
                assert(std::size(last_flanking_bases) == std::size(results));
                if (!std::empty(results) && !std::empty(last_flanking_bases) &&
                    (position < (results.back().end() - last_flanking_bases.back()))) {
                    position = results.back().end() - last_flanking_bases.back();
                    continue;
                }

                int64_t curr_flanking_bases = 0;
                int64_t chunk_start_pos = 0;
                int64_t chunk_start_idx = 0;
                int64_t chunk_end_idx = 0;
                for (curr_flanking_bases = flanking_bases;
                     curr_flanking_bases >= MIN_FLANKING_BASES; --curr_flanking_bases) {
                    chunk_start_pos = position - curr_flanking_bases;
                    chunk_start_idx = searchsorted_left(sample.positions_major, chunk_start_pos);
                    chunk_end_idx = chunk_start_idx + chunk_len;

                    if (chunk_start_idx >= std::ssize(sample.positions_major)) {
                        // This shouldn't be possible, but need to check the bounds.
                        std::ostringstream oss;
                        oss << "Tried to create chunk from chunk_start_pos = " << chunk_start_pos
                            << " on seq_id = " << sample.seq_id
                            << " but the position could not be found in this sample! "
                               "chunk_start_idx = "
                            << chunk_start_idx << ", sample: " << sample;
                        throw std::runtime_error{oss.str()};
                    }

                    // TODO: Handle this properly. E.g. Create an overlapping large chunk at the end.
                    if (chunk_end_idx >= std::ssize(sample.positions_major)) {
                        DORADO_LOG(
                                warn,
                                "Tried to create chunk but chunk exceeds sample length. "
                                "chunk_start_pos = {}, chunk_start_idx = {}, chunk_end_idx = {}, "
                                "positions_major.size() = {}, seq_id = {}. Stopping. Sample: {}",
                                chunk_start_pos, chunk_start_idx, chunk_end_idx,
                                std::ssize(sample.positions_major), sample.seq_id, chunk_end_idx,
                                secondary::sample_to_string(sample));
                        stop_chunking = true;
                        break;
                    }

                    const int64_t chunk_end_pos = sample.positions_major[chunk_end_idx];

                    if (position <= (chunk_end_pos - curr_flanking_bases)) {
                        break;
                    }
                }
                if (stop_chunking) {
                    break;
                }
                if (curr_flanking_bases < MIN_FLANKING_BASES) {
                    DORADO_LOG(warn,
                               "Could not create chunk around position = {} with more than {} "
                               "bases of flanking context. Skipping the rest of this candidate.",
                               position, MIN_FLANKING_BASES);
                    break;
                }

                // Extract the chunk around this position.
                results.emplace_back(slice_sample(sample, chunk_start_idx, chunk_end_idx, true));
                last_flanking_bases.emplace_back(curr_flanking_bases);

#ifdef DEBUG_POLISH_SPLIT_SAMPLES_AROUND_POSITIONS
                DORADO_LOG(debug,
                           "[split_samples_around_positions] Created a chunk around position = {}. "
                           "chunk_start_pos = {}, chunk_start_idx = {}, chunk_end_idx = {}, "
                           "curr_flanking_bases = {}. Chunk sample: {}",
                           position, chunk_start_pos, chunk_start_idx, chunk_end_idx,
                           curr_flanking_bases, secondary::sample_to_string(results.back()));
#endif

                // Continue from the first position not covered by this chunk.
                position = std::max(position + 1, results.back().end() - curr_flanking_bases);
            }
            if (stop_chunking) {
                break;
            }
        }

        all_results.insert(std::end(all_results), std::make_move_iterator(std::begin(results)),
                           std::make_move_iterator(std::end(results)));
        sample = {};
    }

    return all_results;
}

std::vector<secondary::Sample> split_samples_tiled_with_candidates(
        std::vector<secondary::Sample> samples,
        const secondary::IntervalTreesInt64Map& candidate_trees,
        const int64_t chunk_len,
        const int64_t chunk_overlap,
        const bool ext_flanks,          // Control extension heuristic.
        const int64_t ext_major_bases,  // Check this many major positions to trigger.
        const int64_t ext_min_cov,      // Minimum absolute coverage to trigger the heuristic.
        const double ext_cov_frac       // Minimum coverage fraction to trigger the heuristic.
) {
    if (chunk_len <= 0) {
        throw std::runtime_error("Wrong chunk_len. chunk_len = " + std::to_string(chunk_len));
    }
    if ((chunk_overlap < 0) || (chunk_overlap > chunk_len)) {
        throw std::runtime_error(
                "Wrong chunk_overlap length. chunk_len = " + std::to_string(chunk_len) +
                ", chunk_overlap = " + std::to_string(chunk_overlap));
    }

    enum class WindowUsedStatus {
        UNUSED,
        SELECTED_BY_CANDIDATES,
        LEFT_EXT_INSERTIONS,
        LEFT_EXT_DELETIONS,
        RIGHT_EXT_INSERTIONS,
        RIGHT_EXT_DELETIONS,
    };

    const auto has_candidates = [](const secondary::IntervalTreeInt64& tree,
                                   const secondary::Sample& sample, const int64_t start_idx,
                                   const int64_t end_idx) {
        if ((start_idx < 0) || (end_idx <= 0) || (start_idx >= end_idx) ||
            (end_idx > std::ssize(sample.positions_major))) {
            return false;
        }
        const int64_t start = sample.positions_major[start_idx];
        const int64_t end = sample.positions_major[end_idx - 1];  // Inclusive for IntervalTree.
        const std::vector<interval_tree::Interval<int64_t, int64_t>> positions =
                tree.findOverlapping(start, end);
        return !std::empty(positions);
    };

    const auto check_excess_deletions = [](const secondary::Sample& sample, const int64_t start_idx,
                                           const int64_t end_idx, const bool reverse,
                                           const int64_t major_bases, const double cov_fraction,
                                           const int64_t min_abs_cov) {
        /// @brief Returns true if any of the first/last `major_bases` major positions have many deletion
        ///         counts (above `max(cov * cov_fraction, min_abs_cov)`).
        static constexpr int8_t DEL_VAL = 5;  // Value representing deletion in base channel.

        // Find maximum non-padded coverage of this sample.
        const int64_t cov = sample.find_max_depth(start_idx, end_idx);
        const int64_t min_count = std::max(min_abs_cov, static_cast<int64_t>(cov * cov_fraction));

        if (reverse) {
            for (int64_t pos = start_idx, num_major = 0; pos < end_idx; ++pos) {
                if (sample.positions_minor[pos] > 0) {
                    continue;
                }
                ++num_major;
                if (num_major > major_bases) {
                    break;
                }
                const at::Tensor pos_slice = sample.features.index({pos});
                const at::Tensor bases = pos_slice.index({at::indexing::Slice(), 0});
                const at::Tensor mask = bases.eq(DEL_VAL);
                const int64_t count = mask.sum().item<int64_t>();
                if (count >= min_count) {
                    return true;
                }
            }
        } else {
            for (int64_t pos = (end_idx - 1), num_major = 0; pos >= start_idx; --pos) {
                if (sample.positions_minor[pos] > 0) {
                    continue;
                }
                ++num_major;
                if (num_major > major_bases) {
                    break;
                }
                const at::Tensor pos_slice = sample.features.index({pos});
                const at::Tensor bases = pos_slice.index({at::indexing::Slice(), 0});
                const at::Tensor mask = bases.eq(DEL_VAL);
                const int64_t count = mask.sum().item<int64_t>();
                if (count >= min_count) {
                    return true;
                }
            }
        }
        return false;
    };

    const auto extend_left = [&check_excess_deletions, &ext_major_bases, &ext_cov_frac,
                              &ext_min_cov](const secondary::Sample& sample,
                                            const std::vector<std::pair<int64_t, int64_t>>& windows,
                                            const int32_t start_window,
                                            std::span<WindowUsedStatus> window_used) {
        // If the current window is unused or an extension, do not extend.
        if (window_used[start_window] != WindowUsedStatus::SELECTED_BY_CANDIDATES) {
            return;
        }

        if (start_window == 0) {
            return;
        }

        // Get the last major coordinate of the starting window.
        const auto [initial_start_idx, initial_end_idx] = windows[start_window];
        const int64_t initial_first_major = sample.positions_major[initial_start_idx];

        int64_t j = start_window - 1;

        // Extend left due to insertion columns on the left edge until the major coord changes.
        int64_t next_idx = initial_start_idx - 1;
        for (; (j >= 0 && (next_idx >= 0)); --j) {
            if (window_used[j] != WindowUsedStatus::UNUSED) {
                break;
            }
            const auto [start_idx, end_idx] = windows[j];
            const int64_t next_major = sample.positions_major[next_idx];
            if (next_major < initial_first_major) {
                break;
            }
            next_idx = start_idx - 1;
            window_used[j] = WindowUsedStatus::LEFT_EXT_INSERTIONS;
        }

        // Extend further if there are flanking deletions.
        for (; j >= 0; --j) {
            if (window_used[j] != WindowUsedStatus::UNUSED) {
                break;
            }
            const auto [start_idx, end_idx] = windows[j + 1];
            const bool is_deletion = check_excess_deletions(
                    sample, start_idx, end_idx, true, ext_major_bases, ext_cov_frac, ext_min_cov);
            if (!is_deletion) {
                break;
            }
            window_used[j] = WindowUsedStatus::LEFT_EXT_DELETIONS;
        }
    };

    const auto extend_right = [&check_excess_deletions, &ext_major_bases, &ext_cov_frac,
                               &ext_min_cov](
                                      const secondary::Sample& sample,
                                      const std::vector<std::pair<int64_t, int64_t>>& windows,
                                      const int32_t start_window,
                                      std::span<WindowUsedStatus> window_used) {
        // If the current window is unused or an extension, do not extend.
        if (window_used[start_window] != WindowUsedStatus::SELECTED_BY_CANDIDATES) {
            return;
        }

        if (start_window == (std::ssize(window_used) - 1)) {
            return;
        }

        // Get the last major coordinate of the starting window.
        const auto [initial_start_idx, initial_end_idx] = windows[start_window];
        const int64_t initial_last_major = sample.positions_major[initial_end_idx - 1];

        const int64_t num_pos = std::ssize(sample.positions_major);

        int64_t j = start_window + 1;

        // Extend insertions right until we reach a window with MAJOR position > initial_last_major.
        // Reason - this handles 2 cases:
        //      1. Last position of the `start_window` is a minor position, in which case
        //          we need to extend the window right. If the window to the right begins
        //          with the same major coordinate (also a minor position), take it.
        //          If the next window begins on a larger major, that means we saw all
        //          insertion columns for the initial_last_major coordinate.
        //      2. Last position of the `start_window` is a major position, but it is followed
        //          by minor coordinates in the next window. Due to pure chance, we ended up here
        //          by cutting on the major column. In this case, the same heuristic applies - the
        //          next window should have the first major position > initial_last_major to ensure all
        //          insertions are covered.
        int64_t next_idx = initial_end_idx;
        for (; (j < std::ssize(window_used)) && (next_idx < num_pos); ++j) {
            if (window_used[j] != WindowUsedStatus::UNUSED) {
                break;
            }
            const auto [start_idx, end_idx] = windows[j];
            const int64_t next_major = sample.positions_major[next_idx];
            if (next_major > initial_last_major) {
                break;
            }
            next_idx = end_idx;
            window_used[j] = WindowUsedStatus::RIGHT_EXT_INSERTIONS;
        }

        // Extend further if there are flanking deletions.
        for (; j < std::ssize(window_used); ++j) {
            if (window_used[j] != WindowUsedStatus::UNUSED) {
                break;
            }
            const auto [start_idx, end_idx] = windows[j - 1];
            const bool is_deletion = check_excess_deletions(
                    sample, start_idx, end_idx, false, ext_major_bases, ext_cov_frac, ext_min_cov);
            if (!is_deletion) {
                break;
            }
            window_used[j] = WindowUsedStatus::RIGHT_EXT_DELETIONS;
        }
    };

    std::vector<secondary::Sample> results;
    results.reserve(std::size(samples));

    // Retained chunks own cloned tensors, so release each source after it has been processed.
    for (auto& sample : samples) {
        const int64_t sample_len = static_cast<int64_t>(std::size(sample.positions_major));

        // Get the interval tree of candidates for this sequence ID.
        const auto it_seq_id = candidate_trees.find(sample.seq_id);
        if (it_seq_id == std::cend(candidate_trees)) {
            DORADO_LOG(debug, "Cannot find seq_id = {} in candidate_trees! Sample: {}",
                       sample.seq_id, secondary::sample_to_string(sample));
            sample = {};
            continue;
        }
        const auto& tree = it_seq_id->second;

        if (sample_len <= chunk_len) {
            if (has_candidates(tree, sample, 0, sample_len)) {
                results.emplace_back(std::move(sample));
            } else {
                sample = {};
            }
            continue;
        }

        const int64_t step = chunk_len - chunk_overlap;

        // Create window coordinates.
        std::vector<std::pair<int64_t, int64_t>> windows;
        {
            windows.reserve((sample_len - chunk_len) / chunk_overlap);
            int64_t end = 0;
            for (int64_t start = 0; start < (sample_len - chunk_len + 1); start += step) {
                end = start + chunk_len;
                windows.emplace_back(start, end);
            }
            if (end < sample_len) {
                const int64_t start = sample_len - chunk_len;
                end = sample_len;
                windows.emplace_back(start, end);
            }
        }

        // Find windows with candidates.
        std::vector<WindowUsedStatus> window_used(std::size(windows), WindowUsedStatus::UNUSED);
        for (int64_t i = 0; i < std::ssize(windows); ++i) {
            const auto [start, end] = windows[i];
            if (has_candidates(tree, sample, start, end)) {
                window_used[i] = WindowUsedStatus::SELECTED_BY_CANDIDATES;
            }
        }

        // Heuristic to include neighboring windows if there are possible
        // deletions at the flanks.
        if (ext_flanks) {
            for (int64_t i = 0; i < std::ssize(windows); ++i) {
                extend_right(sample, windows, i, window_used);
                extend_left(sample, windows, i, window_used);
            }
        }

        // Slice out selected windows to return.
        for (int64_t i = 0; i < std::ssize(windows); ++i) {
            if (window_used[i] != WindowUsedStatus::UNUSED) {
                const auto [start, end] = windows[i];
                results.emplace_back(slice_sample(sample, start, end, true));
            }
        }
        sample = {};
    }

    return results;
}

std::vector<secondary::Interval64> find_sample_intervals(const secondary::Sample& sample,
                                                         const bool split_on_gaps,
                                                         const int64_t min_depth) {
    sample.validate();

    if (std::empty(sample.positions_major)) {
        return {};
    }

    const std::span<int64_t> depth(sample.depth.data_ptr<int64_t>(),
                                   static_cast<size_t>(sample.depth.size(0)));

    std::vector<secondary::Interval64> intervals;
    int64_t interval_start = -1;

    for (int64_t i = 0; i < std::ssize(sample.positions_major); ++i) {
        const bool is_depth_good = (min_depth <= 0) || (depth[i] >= min_depth);
        const bool has_gap_before =
                split_on_gaps && (i > 0) &&
                ((sample.positions_major[i] - sample.positions_major[i - 1]) > 1);

        if (!is_depth_good) {
            if (interval_start >= 0) {
                intervals.emplace_back(secondary::Interval64{interval_start, i});
                interval_start = -1;
            }
            continue;
        }

        if (interval_start < 0) {
            interval_start = i;
            continue;
        }

        if (has_gap_before) {
            intervals.emplace_back(secondary::Interval64{interval_start, i});
            interval_start = i;
        }
    }

    if (interval_start >= 0) {
        intervals.emplace_back(
                secondary::Interval64{interval_start, std::ssize(sample.positions_major)});
    }

    return intervals;
}

std::vector<secondary::Sample> split_sample_on_intervals(
        const secondary::Sample& sample,
        const std::vector<secondary::Interval64>& intervals) {
    if (std::empty(intervals)) {
        return {};
    }

    // Reusable.
    const std::vector<std::string> placeholder_ids =
            placeholder_read_ids(std::ssize(sample.read_ids_left));

    std::vector<secondary::Sample> ret;
    for (int64_t i = 0; i < std::ssize(intervals); ++i) {
        const secondary::Interval64 iv = intervals[i];

        if ((iv.start < 0) || (iv.end <= iv.start) ||
            (iv.end > std::ssize(sample.positions_major))) {
            throw std::runtime_error(
                    "Invalid sample interval for splitting. interval = [" +
                    std::to_string(iv.start) + ", " + std::to_string(iv.end) +
                    "), sample_len = " + std::to_string(std::size(sample.positions_major)));
        }

        // Generate the sample slice. This propagates read_ids_left and read_ids_right
        // from the original samples.
        Sample new_sample = slice_sample(sample, iv.start, iv.end);

        ret.emplace_back(std::move(new_sample));
    }

    return ret;
}

std::vector<secondary::Sample> split_sample_on_discontinuities(const secondary::Sample& sample) {
    const std::vector<secondary::Interval64> intervals = find_sample_intervals(sample, true, 0);
    if ((std::size(intervals) == 1) && (intervals.front().start == 0) &&
        (intervals.front().end == std::ssize(sample.positions_major))) {
        return {sample};
    }
    return split_sample_on_intervals(sample, intervals);
}

}  // namespace dorado::secondary
