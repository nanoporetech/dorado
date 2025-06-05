#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>

namespace dorado::secondary {

/**
 * \brief Parses a Kadayashi binary file with phasing information.
 *          The file structure has a lookup table and allows for random access to a specified region.
 * \param in_haplotag_bin_fn Path to an input Kadayashi binary phasing file.
 * \param chrom Chromosome name to fetch the data for.
 * \param chrom_start Start coordinate of the chromosome region to fetch. Zero based.
 * \param chrom_end End coordinate of the chromosome region to fetch. Non-inclusive.
 */
std::unordered_map<std::string, int32_t> query_bin_file_get_qname2tag(
        const std::filesystem::path & in_haplotag_bin_fn,
        const std::string & chrom,
        const int64_t chrom_start,
        const int64_t chrom_end);

}  // namespace dorado::secondary
