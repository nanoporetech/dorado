#include <map>
#include <string>
#include <vector>

namespace dorado::utils {
// Given a path to a space-delimited csv in `tempate_id complement_id` format,
// returns a map of template_id to  complement_id
std::map<std::string, std::string> load_pairs_file(std::string pairs_file);

// Compute reverse complement of a nucleotide sequence
void reverse_complement(std::vector<char>& sequence);

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
}  // namespace dorado::utils