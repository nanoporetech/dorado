#pragma once

#include <cstdint>
#include <string>
#include <vector>

// medaka-style feature data
struct ReadAlignmentData {
public:
    ReadAlignmentData(int32_t n_pos_,
                      int32_t n_reads_,
                      int32_t buffer_pos_,
                      int32_t buffer_reads_,
                      int32_t extra_featlen_,
                      int32_t fixed_size_);

    void resize_cols(const int64_t new_buffer_cols);

    void resize_num_reads(const int64_t new_buffer_reads);

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

// /** Prints a pileup data structure.
//  *
//  *  @param pileup a pileup counts structure.
//  *  @returns void
//  *
//  */
// void print_read_aln_data(const ReadAlignmentData& pileup);

/** Generates medaka-style feature data in a region of a bam.
 *
 *  @param region 1-based region string.
 *  @param bam_file input aligment file.
 *  @param num_dtypes number of datatypes in bam.
 *  @param dtypes prefixes on query names indicating datatype.
 *  @param tag_name by which to filter alignments
 *  @param tag_value by which to filter data
 *  @param keep_missing alignments which do not have tag.
 *  @param read_group used for filtering.
 *  @param min_mapQ mininimum mapping quality.
 *  @param row_per_read place each new read on a new row.
 *  @param include_dwells whether to include dwells channel in features.
 *  @param include_haplotype whether to include haplotag channel in features.
 *  @param max_reads maximum allowed read depth.
 *  @returns a pileup counts data pointer.
 *
 *  The return value can be freed with destroy_read_aln_data.
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
 *  If row_per_read is False, new reads will be placed in the first row where
 *  there previous read has terminated, if one exists.
 */
ReadAlignmentData calculate_read_alignment(BamFile &bam_file,
                                           const std::string &chr_name,
                                           const int64_t start,  // Zero-based.
                                           const int64_t end,    // Non-inclusive.
                                           const int64_t num_dtypes,
                                           const std::vector<std::string> &dtypes,
                                           const std::string &tag_name,
                                           const int32_t tag_value,
                                           const bool keep_missing,
                                           const char *read_group,
                                           const int32_t min_mapq,
                                           const bool row_per_read,
                                           const bool include_dwells,
                                           const bool include_haplotype,
                                           const int32_t max_reads);
