#pragma once

#include "secondary/interval.h"
#include "types.h"
#include "utils/cigar.h"
#include "utils/overlap.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace dorado {
struct CorrectionAlignments;
}

namespace dorado::correction {

struct CigarPoint {
    int32_t idx = 0;
    int32_t offset = 0;
    int32_t tpos = 0;
    int32_t qpos = 0;
};

bool extract_windows(std::vector<std::vector<OverlapWindow>>& windows,
                     const CorrectionAlignments& alignments,
                     int window_size);

/**
 * \brief Splits alignments into windows with maximum length of the pileup matrix limited to a specific
 *          value (max_num_columns) to ensure stable memory consumption during inference.
 *
 *          Given a set of query-to-target alignments, this function produces a vector of windows
 *          along the target. Each window contains zero or more query alignment chunks.
 *          The output OverlapWindow objects specify the exact window start/end coordinates,
 *          alignment target start/end coordinates, and start/end CIGAR operations needed to
 *          extract the alignments for any window.
 *
 *          If `max_num_columns` is set to a very large number, this function will
 *          produce results similar to the legacy `extract_windows`. However, `max_num_columns`
 *          provides control over the maximum number of pileup columns (M+I+D)
 *          in a window and thus limits the window size in case of large insertions.
 *
 * \param windows Return vector of windows. Outer dimension represents the target windows, sorted
 *                  by the target start coordinate. Inner dimension is a vector of alignment chunks
 *                  which fall into that window. Any inner vector can be empty if there are no
 *                  alignments there.
 * \param alignments Input alignments to be split into windows. Also used to determine
 *                      window boundaries.
 * \param window_size Window size in the number of target bases, e.g. 4096.
 * \param max_num_columns Maximum number of pileup columns in any given window. Pileup
 *                          columns also include insertion bases (M+I+D), so the width
 *                          of a tensor can otherwise grow drastically in some regions. This
 *                          heuristic will split a window even if it has < 4096 bp of target
 *                          sequence, to maintain the maximum number of pileup columns and limit
 *                          the memory consumption.
 * \returns true if windows were successfully generated, false otherwise.
 */
std::vector<secondary::Interval> extract_limited_windows(
        std::vector<std::vector<OverlapWindow>>& windows,
        const CorrectionAlignments& alignments,
        int32_t window_size,
        int32_t max_num_columns);

std::vector<std::pair<int32_t, OverlapWindow>> split_alignment(
        const utils::Overlap& overlap,
        const std::vector<CigarOp>& cigar,
        const std::vector<secondary::Interval>& win_intervals,
        int32_t aln_id,
        bool legacy_qstart,
        bool custom_initial_point,
        int32_t initial_cig_id,
        int32_t initial_cig_offset,
        int32_t initial_tstart,
        int32_t initial_qstart);

/// @brief Takes a set of alignments and a vector of window intervals, and finds chunks of the alignments
///         which fall within those windows.
/// @param alignments Input alignments.
/// @param win_intervals Window intervals. They do not have to be adjacent, but they have to be sorted.
/// @param window_size Window size in target coordinates, used for heuristic filtering.
/// @param aln_ids Optionally process only the alignments listed here. If empty, all alignments will be processed.
/// @param cigar_points Optional. If empty, all alignments will be processed from the beginning. This is useful for incremental processing.
/// @return A vector of length equal to win_intervals, where each element contains zero or more alignment chunks which fall within these coordinates.
std::vector<std::vector<OverlapWindow>> split_alignments_into_windows(
        const CorrectionAlignments& alignments,
        const std::vector<secondary::Interval>& win_intervals,
        const int32_t window_size,
        const std::vector<int32_t>& aln_ids,
        const std::vector<CigarPoint>& cigar_points);

}  // namespace dorado::correction
