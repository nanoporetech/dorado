#pragma once

#include "secondary/bam_file.h"

#include <cstdint>
#include <string>
#include <vector>

namespace dorado::secondary {

// medaka-style feature data
class ReadAlignmentData {
public:
    ReadAlignmentData(const int32_t n_pos_,
                      const int32_t n_reads_,
                      const int32_t buffer_pos_,
                      const int32_t buffer_reads_,
                      const int32_t extra_featlen_,
                      const int32_t fixed_size_);

    void resize_cols(const int32_t new_buffer_cols);

    void resize_num_reads(const int32_t new_buffer_reads);

    int32_t buffer_pos;
    int32_t buffer_reads;
    int32_t num_dtypes;
    int32_t n_pos;
    int32_t n_reads;
    int32_t featlen;
    std::vector<int8_t> matrix;
    std::vector<int64_t> major;
    std::vector<int64_t> minor;
    std::vector<std::string> read_ids_left;
    std::vector<std::string> read_ids_right;
};

/**
 * \brief Generates medaka-style feature data in a region of a BAM.
 * \param bam_file Input aligment file.
 * \param chr_name Name of the input sequence for the queried region.
 * \param start Start coordinate of the region. Zero-based.
 * \param end End coordinate of the region. Non-inclusive.
 * \param num_dtypes Number of datatypes in bam.
 * \param dtypes Prefixes on query names indicating datatype.
 * \param tag_name Tag name by which to filter alignments.
 * \param tag_value Tag value by which to filter data. Only int supported.
 * \param keep_missing Keeps alignments which do not have the tag specified with tag_name.
 * \param read_group Used for filtering.
 * \param min_mapq Mininimum mapping quality.
 * \param row_per_read Place each new read on a new row.
 * \param include_dwells Include dwells channel in features.
 * \param include_haplotype Include haplotag channel in features.
 * \param max_reads Maximum allowed read depth.
 * \returns ReadAlignmentData object which contains the features, positions and read IDs.
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
 *
 *  If row_per_read is false, new reads will be placed in the first row where
 *  there previous read has terminated, if one exists.
 */
ReadAlignmentData calculate_read_alignment(secondary::BamFile &bam_file,
                                           const std::string &chr_name,
                                           const int64_t start,  // Zero-based.
                                           const int64_t end,    // Non-inclusive.
                                           const int64_t num_dtypes,
                                           const std::vector<std::string> &dtypes,
                                           const std::string &tag_name,
                                           const int32_t tag_value,
                                           const bool keep_missing,
                                           const std::string &read_group,
                                           const int32_t min_mapq,
                                           const bool row_per_read,
                                           const bool include_dwells,
                                           const bool include_haplotype,
                                           const int32_t max_reads);

}  // namespace dorado::secondary