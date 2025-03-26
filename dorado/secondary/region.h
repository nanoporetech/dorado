#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

namespace dorado::secondary {

struct RegionInt {
    int32_t seq_id = -1;
    int64_t start = 0;
    int64_t end = -1;
};

struct Region {
    std::string name;
    int64_t start = 0;
    int64_t end = -1;
};

std::ostream& operator<<(std::ostream& os, const Region& region);

bool operator<(const Region& l, const Region& r);

std::string region_to_string(const Region& region);

/**
 * \brief Parses a Htslib-style region from an input string. The input Htslib-style
 *          region uses a 1-based start coordinate and an inclusive end coordinate, but the
 *          returned Region object is converted into zero-based and non-inclusive end coordinate.
 *
 *          The input region can be formatted in one of the following ways:
 *              - "chr" - only the sequence name is specified.
 *              - "chr:start" - sequence name and the start coordinate are specified.
 *              - "chr:start-end" - full region specification.
 *
 *          In case start/end are not specified, they are set to -1.
 */
Region parse_region_string(const std::string& region);

/**
 * \brief Parses regions either from a BED-like file on disk, or from a comma
 *          separated list of Htslib-style regions in the given string.
 *          In case of the latter, regions are converted into zero-based non-inclusive
 *          coordinates.
 */
std::vector<Region> parse_regions(const std::string& regions_arg);

/**
 * \brief Checks if any region overlaps any of the other regions and that all coordinates
 *          are within bounds of the sequence lengths.
 */
void validate_regions(const std::vector<Region>& regions,
                      const std::vector<std::pair<std::string, int64_t>>& seq_lens);

}  // namespace dorado::secondary
