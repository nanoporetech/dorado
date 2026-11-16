#pragma once

#include "secondary/common/interval.h"
#include "secondary/common/interval_tree_types.h"
#include "secondary/features/feature_columns.h"

#include <ATen/core/TensorBody.h>

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace dorado::secondary {

struct Sample {
    int32_t seq_id = -1;
    at::Tensor features;
    std::vector<int64_t> positions_major;
    std::vector<int64_t> positions_minor;
    at::Tensor depth;
    std::vector<std::string> read_ids_left;
    std::vector<std::string> read_ids_right;
    std::optional<at::Tensor> draft_seq{std::nullopt};

    int64_t start() const { return (std::empty(positions_major) ? -1 : (positions_major.front())); }

    int64_t end() const {
        return (std::empty(positions_major) ? -1 : (positions_major.back() + 1));
    }

    std::pair<int64_t, int64_t> get_position(const int64_t idx) const {
        if ((idx < 0) || (idx >= static_cast<int64_t>(std::size(positions_major)))) {
            return {-1, -1};
        }
        return {positions_major[idx], positions_minor[idx]};
    }

    std::pair<int64_t, int64_t> get_last_position() const {
        return get_position(static_cast<int64_t>(std::size(positions_major)) - 1);
    }

    int64_t find_max_depth(int64_t start_idx, int64_t end_idx) const;

    void validate() const;

    /**
     * \brief Select a single feature channel from this->features.
     * \param feature_column Channel to select
     * \param feature_column_map Map used to find the channel index
     * \return A contiguous copy of the feature selected channel, shape=(positions, reads)
     */
    at::Tensor select_feature_channel(const FeatureColumns& feature_column,
                                      const FeatureColumnMap& feature_column_map) const;
};

std::vector<std::string> placeholder_read_ids(const int64_t n);

Sample slice_sample(const Sample& sample,
                    const int64_t idx_start,
                    const int64_t idx_end,
                    const bool clone);

Sample slice_sample(const Sample& sample, const int64_t idx_start, const int64_t idx_end);

void debug_print_sample(std::ostream& os,
                        const Sample& sample,
                        int64_t start /*= 0*/,
                        int64_t end /*= -1 */,
                        bool debug /*= false */);

std::ostream& operator<<(std::ostream& os, const Sample& sample);

std::string sample_to_string(const Sample& sample);

/**
 * \brief Takes an input sample and splits it bluntly into overlapping windows.
 *          Splitting is implemented to match Medaka, where a simple sliding window is used to create smaller samples.
 *          In case of a short trailing portion (shorter than chunk_len), a potentially large overlap is produced to
 *          cover this region instead of just outputing the small chunk.
 */
std::vector<secondary::Sample> split_samples(std::vector<Sample> samples,
                                             const int64_t chunk_len,
                                             const int64_t chunk_overlap);

std::vector<Sample> split_samples_around_positions(
        std::vector<Sample> samples,
        const secondary::IntervalTreesInt64Map& candidate_trees,
        const int64_t chunk_len,
        const int64_t flanking_bases);

std::vector<secondary::Sample> split_samples_tiled_with_candidates(
        std::vector<secondary::Sample> samples,
        const secondary::IntervalTreesInt64Map& candidate_trees,
        const int64_t chunk_len,
        const int64_t chunk_overlap,
        const bool ext_flanks,          // Control extension heuristic.
        const int64_t ext_major_bases,  // Check this many major positions to trigger.
        const int64_t ext_min_cov,      // Minimum absolute coverage to trigger the heuristic.
        const double ext_cov_frac       // Minimum coverage fraction to trigger the heuristic.
);

/**
 * \brief Finds contiguous intervals of sample columns which should be kept for processing.
 *          Low-coverage columns are excluded and coordinate gaps split adjacent intervals.
 */
std::vector<secondary::Interval64> find_sample_intervals(const secondary::Sample& sample,
                                                         const bool split_on_gaps,
                                                         const int64_t min_depth);

std::vector<secondary::Sample> split_sample_on_intervals(
        const secondary::Sample& sample,
        const std::vector<secondary::Interval64>& intervals);

/**
 * \brief If the input sample coordinates (positions_major) have gaps,
 *          this function splits the sample on those gaps and produces
 *          one or more samples in the output.
 *          When possible, input data is moved to the output, and that is
 *          why the inpunt is not const.
 */
std::vector<secondary::Sample> split_sample_on_discontinuities(const secondary::Sample& sample);

}  // namespace dorado::secondary
