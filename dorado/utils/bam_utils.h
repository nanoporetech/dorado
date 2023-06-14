#pragma once
#include "htslib/sam.h"
#include "read_pipeline/ReadPipeline.h"

#include <map>
#include <string>
#include <unordered_map>

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

void add_rg_hdr(sam_hdr_t* hdr, const std::unordered_map<std::string, ReadGroup>& read_groups);

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

}  // namespace dorado::utils
