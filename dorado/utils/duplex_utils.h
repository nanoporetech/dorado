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
std::map<std::string, std::string> load_pairs_file(std::string pairs_file);

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
void preprocess_quality_scores(std::vector<uint8_t>& quality_scores, int pool_window = 5);

std::unordered_set<std::string> get_read_list_from_pairs(
        std::map<std::string, std::string> template_complement_map);

/**
 * Returns the stereo model name based on the presence of a specific substring in the input model.
 *
 * This function checks if the input model string contains the substring "4.2". If it does,
 * the function returns the model name for the 5 kHz stereo model. Otherwise, it returns the model
 * name for the 4 kHz stereo.
 *
 * @param model A string representing the input model.
 * @return The corresponding stereo model name as a string.
 *
 * Note: This approach is not very clean but is necessary because the simplex model metadata
 * does not include the sampling rate.
 */
const std::string get_stereo_model_name(const std::string& simplex_model_name,
                                        uint16_t data_sample_rate);

}  // namespace dorado::utils
