#pragma once

#include "sample.h"
#include "secondary/region.h"

#include <cstdint>
#include <iosfwd>
#include <optional>
#include <vector>

namespace dorado::secondary {

struct TrimInfo {
    int64_t start = 0;
    int64_t end = -1;
    bool heuristic = false;
};

/**
 * \brief Checks the start/end coordinates of the provided TrimInfo object.
 *          If they are not valid (i.e. start < 0, end <= 0 or start >= end), then
 *          this function returns false.
 */
bool is_trim_info_valid(const TrimInfo& t);

/**
 * \brief Checks the start/end coordinates of the provided TrimInfo object.
 *          If they are not valid (i.e. start < 0, end <= 0 or start >= end), then
 *          this function returns false.
 *
 *          Additionally, checks if the coordinates are smaller than `len`
 *          (i.e. start < len, end <= len).
 */
bool is_trim_info_valid(const TrimInfo& t, int64_t len);

/**
 * \brief Finds the trimming coordinates for each sample, so that they can be spliced directly.
 *          If a `region` is provided, then the samples will also be trimmed to fit into that region.
 *          Samples completely filtered by the `region` will have TrimInfo{-1, -1, false}.
 *
 * \param samples A vector of samples to trim. Client of this function is responsible for sorting them. Only neighboring samples are compared.
 * \param region Optional region to trim to. After samples are trimmed on neighboring overlaps, region trimming is applied.
 * \return Vector of trimming regions, the same size as the input samples. If a sample is supposed to be completely trimmed out, the start/end coordinates will be set to -1.
 */
std::vector<TrimInfo> trim_samples(const std::vector<Sample>& samples,
                                   const std::optional<const RegionInt>& region);

/**
 * \brief Identical functionality to the above trim_samples, but this one allows for more efficient sample
 *          comparison in case the client code has samples in a permuted order.
 *          In this case, samples do not have to contiguous in memory, only their pointers.
 *
 * \param samples A vector of pointers to samples to trim. Client of this function is responsible for sorting them. Only neighboring samples are compared.
 * \param region Optional region to trim to. After samples are trimmed on neighboring overlaps, region trimming is applied.
 * \return Vector of trimming regions, the same size as the input samples. If a sample is supposed to be completely trimmed out, the start/end coordinates will be set to -1.
 */
std::vector<TrimInfo> trim_samples(const std::vector<const Sample*>& samples,
                                   const std::optional<const RegionInt>& region);

bool operator==(const TrimInfo& lhs, const TrimInfo& rhs);

std::ostream& operator<<(std::ostream& os, const TrimInfo& rhs);

}  // namespace dorado::secondary
