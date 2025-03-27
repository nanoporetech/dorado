#pragma once

#include "medaka_bamiter.h"
#include "secondary/bam_file.h"

#include <array>
#include <cstdint>
#include <iosfwd>
#include <string_view>
#include <vector>

namespace dorado::secondary {

// medaka-style base encoding
static constexpr std::string_view PILEUP_BASES{"acgtACGTdD"};
static constexpr int64_t PILEUP_BASES_SIZE = static_cast<int64_t>(std::size(PILEUP_BASES));
static constexpr int64_t PILEUP_POS_DEL_FWD = 9;  // position of D
static constexpr int64_t PILEUP_POS_DEL_REV = 8;  // position of d

// bam tag used for datatypes
static constexpr std::string_view DATATYPE_TAG{"DT\0"};

// convert 16bit IUPAC (+16 for strand) to PILEUP_BASES index
static constexpr std::array<int32_t, 32> NUM_TO_COUNT_BASE{
        -1, 4, 5, -1, 6, -1, -1, -1, 7, -1, -1, -1, -1, -1, -1, -1,
        -1, 0, 1, -1, 2, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1,
};

class PileupData {
public:
    /**
     * \brief Constructs a pileup data structure.
     * \param n_cols Number of pileup columns.
     * \param buffer_cols Number of pileup columns.
     * \param num_dtypes Number of datatypes in pileup.
     * \param num_homop Maximum homopolymer length to consider.
     * \param fixed_size If not zero data matrix is allocated as fixed_size * n_cols, ignoring other arguments.
     */
    PileupData(const int64_t n_cols_,
               const int64_t buffer_cols_,
               const int64_t num_dtypes_,
               const int64_t num_homop_,
               const int64_t fixed_size_);

    /**
     * \brief Resize the internal buffers of a pileup data structure.
     * \param buffer_cols Number of pileup columns for which to allocate memory.
     */
    void resize_cols(const int64_t buffer_cols);

    int64_t buffer_cols = 0;
    int64_t num_dtypes = 0;
    int64_t num_homop = 0;
    int64_t n_cols = 0;
    std::vector<int64_t> matrix;
    std::vector<int64_t> major;
    std::vector<int64_t> minor;
};

/**
 * \brief Prints a pileup data structure.
 * \param pileup a pileup structure.
 * \param num_dtypes number of datatypes in the pileup.
 * \param dtypes datatype prefix strings.
 * \param num_homop maximum homopolymer length to consider.
 */
void print_pileup_data(std::ostream &os,
                       const PileupData &pileup,
                       const int64_t num_dtypes,
                       const std::vector<std::string> &dtypes,
                       const int64_t num_homop);

/**
 * \brief Generates medaka-style feature data in a region of a BAM.
 * \param bam_file Input aligment file.
 * \param chr_name Name of the input sequence for the queried region.
 * \param start Start coordinate of the region. Zero-based.
 * \param end End coordinate of the region. Non-inclusive.
 * \param num_dtypes Number of datatypes in bam.
 * \param dtypes Prefixes on query names indicating datatype.
 * \param num_homopo Maximum homopolymer length to consider.
 * \param tag_name Tag name by which to filter alignments.
 * \param tag_value Tag value by which to filter data. Only int supported.
 * \param keep_missing Keeps alignments which do not have the tag specified with tag_name.
 * \param weibull_summation Use predefined BAM tags to perform homopolymer partial counts.
 * \param read_group Used for filtering.
 * \param min_mapq Mininimum mapping quality.
 * \returns PileupData object which contains base counts per column.
 *
 * Throws exceptions on errors.
 *
 *  If num_dtypes is 1, dtypes should be empty; all reads in the BAM will be
 *  treated equally. If num_dtypes is not 1, dtypes should be an array of
 *  strings, these strings being prefixes of query names of reads within the
 *  bam file. Any read not matching the prefixes will cause an exception to be thrown.
 *
 *  If tag_name is not empty, alignments are filtered by the (integer) tag value.
 *  When tag_name is given, the behaviour for alignments without the tag is
 *  determined by keep_missing.
 */
PileupData calculate_pileup(secondary::BamFile &bam_file,
                            const std::string &chr_name,
                            const int64_t start,  // Zero-based.
                            const int64_t end,    // Non-inclusive.
                            const int64_t num_dtypes,
                            const std::vector<std::string> &dtypes,
                            const int64_t num_homop,
                            const std::string &tag_name,
                            const int32_t tag_value,
                            const bool keep_missing,
                            const bool weibull_summation,
                            const std::string &read_group,
                            const int32_t min_mapq);

}  // namespace dorado::secondary
