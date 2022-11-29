#include "duplex_utils.h"

#include "torch/torch.h"

#include <algorithm>
#include <fstream>
#include <vector>

namespace dorado::utils {
std::map<std::string, std::string> load_pairs_file(std::string pairs_file) {
    std::ifstream dataFile;
    dataFile.open(pairs_file);

    std::map<std::string, std::string> template_complement_map;

    if (!dataFile.is_open()) {
        throw std::runtime_error("Pairs file does not exist.");
    }
    std::string cell;
    int line = 0;

    std::getline(dataFile, cell);
    while (!dataFile.eof()) {
        char delim = ' ';
        auto delim_pos = cell.find(delim);

        std::string t = cell.substr(0, delim_pos);
        std::string c = cell.substr(delim_pos + 1, delim_pos * 2 - 1);
        template_complement_map[t] = c;

        line++;
        std::getline(dataFile, cell);
    }
    return template_complement_map;
}

void reverse_complement(std::vector<char>& sequence) {
    std::reverse(sequence.begin(), sequence.end());
    std::map<char, char> complementary_nucleotides = {
            {'A', 'T'}, {'C', 'G'}, {'G', 'C'}, {'T', 'A'}};
    std::for_each(sequence.begin(), sequence.end(),
                  [&complementary_nucleotides](char& c) { c = complementary_nucleotides[c]; });
}

std::pair<std::pair<int, int>, std::pair<int, int>> get_trimmed_alignment(
        int num_consecutive_wanted,
        unsigned char* alignment,
        int alignment_length,
        int target_cursor,
        int query_cursor,
        int start_alignment_position,
        int end_alignment_position) {
    int num_consecutive = 0;

    // Find forward trim.
    while (num_consecutive < num_consecutive_wanted) {
        if (alignment[start_alignment_position] != 2) {
            target_cursor++;
        }

        if (alignment[start_alignment_position] != 1) {
            query_cursor++;
        }

        if (alignment[start_alignment_position] == 0) {
            num_consecutive++;
        } else {
            num_consecutive = 0;  //reset counter
        }

        start_alignment_position++;

        if (start_alignment_position >= alignment_length) {
            break;
        }
    }

    target_cursor -= num_consecutive_wanted;
    query_cursor -= num_consecutive_wanted;

    // Find reverse trim
    num_consecutive = 0;
    while (num_consecutive < num_consecutive_wanted) {
        if (alignment[end_alignment_position] == 0) {
            num_consecutive++;
        } else {
            num_consecutive = 0;
        }

        end_alignment_position--;

        if (end_alignment_position < start_alignment_position) {
            break;
        }
    }

    start_alignment_position -= num_consecutive_wanted;
    end_alignment_position += num_consecutive_wanted;

    auto alignment_start_end = std::make_pair(start_alignment_position, end_alignment_position);
    auto query_target_cursors = std::make_pair(query_cursor, target_cursor);

    return std::make_pair(alignment_start_end, query_target_cursors);
}

// Applies a min pool filter to q scores for basespace-duplex algorithm
void preprocess_quality_scores(std::vector<uint8_t>& quality_scores, int pool_window) {
    // Apply a min-pool window to the quality scores
    auto opts = torch::TensorOptions().dtype(torch::kInt8);
    torch::Tensor t =
            torch::from_blob(quality_scores.data(), {1, (int)quality_scores.size()}, opts);
    auto t_float = t.to(torch::kFloat32);
    t.index({torch::indexing::Slice()}) =
            -torch::max_pool1d(-t_float, pool_window, 1, pool_window / 2);
}

}  // namespace dorado::utils