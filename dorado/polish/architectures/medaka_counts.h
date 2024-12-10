#pragma once

#include "medaka_bamiter.h"
#include "polish/bam_file.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string_view>
#include <vector>

namespace dorado::polisher {

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
    /** Constructs a pileup data structure.
     *
     *  @param n_cols number of pileup columns.
     *  @param buffer_cols number of pileup columns.
     *  @param num_dtypes number of datatypes in pileup.
     *  @param num_homop maximum homopolymer length to consider.
     *  @param fixed_size if not zero data matrix is allocated as fixed_size * n_cols, ignoring other arguments.
     *
     *  The return value can be freed with destroy_plp_data.
     *
     */
    PileupData(const int64_t n_cols,
               const int64_t buffer_cols,
               const int64_t num_dtypes,
               const int64_t num_homop,
               const int64_t fixed_size);

    /** Resize the internal buffers of a pileup data structure.
     *
     *  @param buffer_cols number of pileup columns for which to allocate memory.
     *
     */
    void resize_cols(const int64_t buffer_cols);

    int64_t buffer_cols() const { return m_buffer_cols; }
    int64_t num_dtypes() const { return m_num_dtypes; }
    int64_t num_homop() const { return m_num_homop; }
    int64_t n_cols() const { return m_n_cols; }

    // Can't call thse getters and setters just `major()` or `minor` because:
    //  "error: In the GNU C Library, "major" is defined"
    const std::vector<int64_t>& get_matrix() const { return m_matrix; }
    const std::vector<int64_t>& get_major() const { return m_major; }
    const std::vector<int64_t>& get_minor() const { return m_minor; }

    std::vector<int64_t>& get_matrix() { return m_matrix; }
    std::vector<int64_t>& get_major() { return m_major; }
    std::vector<int64_t>& get_minor() { return m_minor; }

    void set_major(std::vector<int64_t> data) { std::swap(m_major, data); }
    void set_minor(std::vector<int64_t> data) { std::swap(m_minor, data); }

    void n_cols(const int64_t val) { m_n_cols = val; }

private:
    int64_t m_buffer_cols = 0;
    int64_t m_num_dtypes = 0;
    int64_t m_num_homop = 0;
    int64_t m_n_cols = 0;
    std::vector<int64_t> m_matrix;
    std::vector<int64_t> m_major;
    std::vector<int64_t> m_minor;
};

/** Prints a pileup data structure.
 *
 *  @param pileup a pileup counts structure.
 *  @param num_dtypes number of datatypes in the pileup.
 *  @param dtypes datatype prefix strings.
 *  @param num_homop maximum homopolymer length to consider.
 *  @returns void
 *
 */
void print_pileup_data(std::ostream& os,
                       const PileupData& pileup,
                       const int64_t num_dtypes,
                       const std::vector<std::string>& dtypes,
                       const int64_t num_homop);

/** Generates medaka-style feature data in a region of a bam.
 *
 *  @param region 1-based region string.
 *  @param bam_file input aligment file.
 *  @param num_dtypes number of datatypes in bam.
 *  @param dtypes prefixes on query names indicating datatype.
 *  @param num_homop maximum homopolymer length to consider.
 *  @param tag_name by which to filter alignments
 *  @param tag_value by which to filter data
 *  @param keep_missing alignments which do not have tag
 *  @param weibull_summation use predefined bam tags to perform homopolymer partial counts.
 *  @param read group used for filtering.
 *  @param mininimum mapping quality for filtering.
 *  @returns a pileup counts data pointer.
 *
 *  The return value can be freed with destroy_plp_data.
 *
 *  If num_dtypes is 1, dtypes should be NULL; all reads in the bam will be
 *  treated equally. If num_dtypes is not 1, dtypes should be an array of
 *  strings, these strings being prefixes of query names of reads within the
 *  bam file. Any read not matching the prefixes will cause exit(1).
 *
 *  If tag_name is not NULL alignments are filtered by the (integer) tag value.
 *  When tag_name is given the behaviour for alignments without the tag is
 *  determined by keep_missing.
 *
 */
PileupData calculate_pileup(BamFile& bam_file,
                            const std::string& seq_name,
                            const int64_t region_start,
                            const int64_t region_end,
                            const int64_t num_dtypes,
                            const std::vector<std::string>& dtypes,
                            const int64_t num_homop,
                            const std::string& tag_name,
                            const int32_t tag_value,
                            const bool keep_missing,
                            const bool weibull_summation,
                            const std::string& read_group,
                            const int32_t min_mapq);

}  // namespace dorado::polisher
