#pragma once
#include "types.h"

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

struct sam_hdr_t;

namespace dorado::utils {

using sq_t = std::vector<std::pair<char*, uint32_t>>;

struct AlignmentOps {
    size_t softclip_start;
    size_t softclip_end;
    size_t matches;
    size_t insertions;
    size_t deletions;
    size_t substitutions;
};

void add_rg_hdr(sam_hdr_t* hdr,
                const std::unordered_map<std::string, ReadGroup>& read_groups,
                const std::vector<std::string>& barcode_kits);

void add_sq_hdr(sam_hdr_t* hdr, const sq_t& seqs);

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
std::map<std::string, std::string> extract_pg_keys_from_hdr(const std::string filename,
                                                            const std::vector<std::string>& keys);

/*
 * Extract the sequence string.
 *
 * @param input_record Record to fetch sequence from..
 * @param seqlen Sequence length.
 * @return Vector of sequence quality.
 */
std::string extract_sequence(bam1_t* input_record, int seqlen);

/*
 * Extract the sequence quality information.
 *
 * @param input_record Record to fetch quality from.
 * @param seqlen Sequence length.
 * @return Vector of sequence quality.
 */
std::vector<uint8_t> extract_quality(bam1_t* input_record, int seqlen);

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

}  // namespace dorado::utils
