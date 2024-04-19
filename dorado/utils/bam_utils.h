#pragma once
#include "barcode_kits.h"
#include "types.h"

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

struct sam_hdr_t;
struct kstring_t;

namespace dorado::utils {

class SampleSheet;

using sq_t = std::vector<std::pair<char*, uint32_t>>;

struct AlignmentOps {
    size_t softclip_start;
    size_t softclip_end;
    size_t matches;
    size_t insertions;
    size_t deletions;
    size_t substitutions;
};

void add_rg_headers(sam_hdr_t* hdr, const std::unordered_map<std::string, ReadGroup>& read_groups);

void add_rg_headers_with_barcode_kit(
        sam_hdr_t* hdr,
        const std::unordered_map<std::string, ReadGroup>& read_groups,
        const std::string& kit_name,
        const barcode_kits::KitInfo& kit_info,
        const std::unordered_map<std::string, std::string>& custom_sequences,
        const utils::SampleSheet* const sample_sheet);

void add_sq_hdr(sam_hdr_t* hdr, const sq_t& seqs);

/// Remove SO tag and any SQ lines from the header.
void strip_alignment_data_from_header(sam_hdr_t* hdr);

/**
 * @brief Retrieves read group information from a SAM/BAM/CRAM file header based on a specified key.
 *
 * This function extracts read group information from a SAM/BAM/CRAM file header
 * and returns a map containing read group IDs and their associated tags for the specified key.
 *
 * @param header A pointer to a valid sam_hdr_t object representing the SAM/BAM/CRAM file header.
 * @param key A null-terminated C-style string representing the tag key to be retrieved for each read group.
 * @return A std::map containing read group IDs as keys and their associated tags for the specified key as values.
 *
 * @throws std::invalid_argument If the provided header is nullunordered_ptr or key is nullptr/empty.
 * @throws std::runtime_error If there are no read groups in the file.
 *
 * Example usage:
 * 
 * samFile *sam_fp = sam_open("example.bam", "r");
 * sam_hdr_t *header = sam_hdr_read(sam_fp);
 * std::map<std::string, std::string> read_group_info = get_read_group_info(header, "DT");
 *
 * for (const auto& rg_pair : read_group_info) {
 *     std::cout << "Read Group ID: " << rg_pair.first << ", Date: " << rg_pair.second << std::endl;
 * }
 */
std::map<std::string, std::string> get_read_group_info(sam_hdr_t* header, const char* key);

/**
 * @brief Calculates the count of various alignment operations (soft clipping, matches, insertions, deletions, and substitutions) in a given BAM record.
 *
 * This function takes a pointer to a bam1_t structure as input and computes the count of soft clipping,
 * matches, insertions, deletions, and substitutions in the alignment using the CIGAR string and MD tag.
 * It returns an AlignmentOps structure containing these counts.
 *
 * @param record Pointer to a bam1_t structure representing a BAM record.
 * @return AlignmentOps structure containing counts of soft clipping, matches, insertions, deletions, and substitutions in the alignment.
 */
AlignmentOps get_alignment_op_counts(bam1_t* record);

/**
 * Extract keys for PG header from BAM header.
 *
 * @param filepath Path to input BAM file.
 * @params keys Vector of keys to parse
 * @return Map of keys to their values
 * @throws An error if a key is requested that doesn't exist.
 */
std::map<std::string, std::string> extract_pg_keys_from_hdr(const std::string& filename,
                                                            const std::vector<std::string>& keys);

/*
 * Extract the sequence string.
 *
 * @param input_record Record to fetch sequence from.
 * @return The sequence bases as a string.
 */
std::string extract_sequence(bam1_t* input_record);

/*
 * Extract the sequence quality information.
 *
 * @param input_record Record to fetch quality from.
 * @return Vector of sequence quality.
 */
std::vector<uint8_t> extract_quality(bam1_t* input_record);

/*
 * Extract the move table from a record, if it exists.
 *
 * @param input_record Record to fetch move table from.
 * @return Tuple where first element is the strice and second element is the vector with moves.
 * An empty vector and stride = 0 are returned if move table doesn't exist.
 */
std::tuple<int, std::vector<uint8_t>> extract_move_table(bam1_t* input_record);

/*
 * Extract mod base tag information from a record.
 *
 * @param input_record Record to fetch mod base information from.
 * @return A tuple where the first element is the modbase tag string, and the
 * second is a vector with modbase probabilities. If no modbase information
 * is found, an empty string and vector are returned respectively.
 */
std::tuple<std::string, std::vector<uint8_t>> extract_modbase_info(bam1_t* input_record);

/*
 * Check that the modified base code supplied is suitable for use in the MM:Z bam tag.
 *
 * @param bam_name mod_bases alphabet entry  
 * @return True if the entry is valid for use (i.e. is single letter or integer ChEBI code)
 */
bool validate_bam_tag_code(const std::string& bam_name);

/*
 * Trim CIGAR string based on position interval for the query sequence.
 *
 * @param n_cigar Number of CIGAR entries
 * @param cigar uint32_t array with CIGAR entries
 * @param interval Position interval to keep after trim
 * @return A vector with uint32_t encoding for CIGAR ops.
 */
std::vector<uint32_t> trim_cigar(uint32_t n_cigar,
                                 const uint32_t* cigar,
                                 const std::pair<int, int>& trim_interval);

/*
 * Calculate how many reference positions are consumed
 * till a specific position on the query based on teh CIGAR
 * string.
 *
 * @param n_cigar Number of CIGAR entries
 * @param cigar uint32_t array of CIGAR entries
 * @param query_pos Query position
 * @return Number of positions consumed in reference
 */
uint32_t ref_pos_consumed(uint32_t n_cigar, const uint32_t* cigar, uint32_t query_pos);

/*
 * Convert a CIGAR uint32_t array to canonical string format.
 *
 * @param n_cigar Number of cigar entries.
 * @param cigar uint32_t array with CIGAR entries
 * @return CIGAR string
 */
std::string cigar2str(uint32_t n_cigar, const uint32_t* cigar);

/*
 * Allocate kstring_t which is already resized to hold 1 MB of data.
 *
 * This is done to work around a Windows DLL cross-heap segmentation fault.
 * The ks_resize/ks_free functions from htslib
 * are inline functions. When htslib is shipped as a DLL, some of these functions
 * are inlined into the DLL code through other htslib APIs. But those same functions
 * also get inlined into dorado code when invoked directly. As a result, it's possible
 * that an htslib APIs resizes a string using the DLL code. But when a ks_free
 * is attempted on it from dorado, there's cross-heap behavior and a segfault occurs.
 *
 * @return kstring_t struct
 */
kstring_t allocate_kstring();

/*
 * Make a copy of the bam record with any alignment data stripped out.
 *
 * @param record BAM record.
 * @param seq New sequence.
 * @param qual New quality scores.
 * 
 * If seq is an empty string, then qual is ignored and the sequence and qualities from the
 * input record will be used. If seq is not empty then it will replace the original sequence
 * in the new object, and qual will replace the original qualities in the new object. This
 * means that if seq is not empty but qual is, then the new object will have nullptr as its
 * quality field. If seq and qual are both non-empty, then of course they must be the same
 * length.
 */
BamPtr new_unmapped_record(const BamPtr& record, std::string seq, std::vector<uint8_t> qual);

/*
 * Remove any alignment related tags from a BAM record.
 *
 * @param record BAM record.
 */
void remove_alignment_tags_from_record(bam1_t* record);

}  // namespace dorado::utils
