#pragma once

#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace dorado::utils {
// Given a path to a space-delimited csv in `tempate_id complement_id` format,
// returns a map of template_id to  complement_id
std::map<std::string, std::string> load_pairs_file(const std::string& pairs_file);

// Returns subset of alignment for which start and end start with  `num_consecutive_wanted` consecutive nucleotides.
std::pair<std::pair<int, int>, std::pair<int, int>> get_trimmed_alignment(
        int num_consecutive_wanted,
        unsigned char* alignment,
        int alignment_length,
        int target_cursor,
        int query_cursor,
        int start_alignment_position,
        int end_alignment_position);

// Applies a min pool filter to q scores for basespace-duplex algorithm
void preprocess_quality_scores(std::vector<uint8_t>& quality_scores);

std::unordered_set<std::string> get_read_list_from_pairs(
        const std::map<std::string, std::string>& template_complement_map);

}  // namespace dorado::utils
