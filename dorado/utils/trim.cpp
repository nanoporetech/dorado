#include "trim.h"

#include <htslib/sam.h>

#include <algorithm>

namespace dorado::utils {

int trim(const torch::Tensor& signal, float threshold, int window_size, int min_elements) {
    const int min_trim = 10;
    const int num_samples = static_cast<int>(signal.size(0)) - min_trim;
    const int num_windows = num_samples / window_size;

    // Access via raw pointers because of torch indexing overhead.
    const auto signal_f32 = signal.to(torch::kFloat32);
    assert(signal_f32.is_contiguous());
    const float* const signal_f32_ptr = signal_f32.data_ptr<float>();

    bool seen_peak = false;
    for (int pos = 0; pos < num_windows; ++pos) {
        const int start = pos * window_size + min_trim;
        const int end = start + window_size;
        assert(start < signal.size(0));
        assert(end <= signal.size(0));  // end is exclusive

        const auto num_large_enough =
                std::count_if(&signal_f32_ptr[start], &signal_f32_ptr[end],
                              [threshold](float elem) { return elem > threshold; });

        if (num_large_enough > min_elements || seen_peak) {
            seen_peak = true;
            if (signal_f32_ptr[end - 1] > threshold) {
                continue;
            }
            if (end >= num_samples) {
                return min_trim;
            } else {
                return end;
            }
        }
    }

    return min_trim;
}

std::string trim_sequence(const std::string& seq, const std::pair<int, int>& trim_interval) {
    int start_pos = trim_interval.first;
    int len = trim_interval.second - start_pos;
    return seq.substr(start_pos, len);
}

std::vector<uint8_t> trim_quality(const std::vector<uint8_t>& qual,
                                  const std::pair<int, int>& trim_interval) {
    if (!qual.empty()) {
        return std::vector<uint8_t>(qual.begin() + trim_interval.first,
                                    qual.begin() + trim_interval.second);
    }
    return {};
}

std::tuple<int, std::vector<uint8_t>> trim_move_table(const std::vector<uint8_t>& move_vals,
                                                      const std::pair<int, int>& trim_interval) {
    std::vector<uint8_t> trimmed_moves;
    int num_positions_trimmed = 0;
    if (!move_vals.empty() && (trim_interval.second > trim_interval.first)) {
        // Start with -1 because as soon as the first move_val==1 is encountered,
        // we have moved to the first base.
        int seq_base_pos = -1;
        for (int i = 0; i < move_vals.size(); i++) {
            auto mv = move_vals[i];
            if (mv == 1) {
                seq_base_pos++;
            }
            if (seq_base_pos >= trim_interval.second) {
                break;
            } else if (seq_base_pos >= trim_interval.first) {
                trimmed_moves.push_back(mv);
            } else {
                num_positions_trimmed++;
            }
        }
    }
    return {num_positions_trimmed, trimmed_moves};
}

std::tuple<std::string, std::vector<int8_t>> trim_modbase_info(
        const std::string& modbase_str,
        const std::vector<int8_t>& modbase_probs,
        const std::pair<int, int>& trim_interval) {
    int start = trim_interval.first;
    int end = trim_interval.second;

    std::string trimmed_modbase_str;
    std::vector<int8_t> trimmed_modbase_probs;
    if (!modbase_str.empty()) {
        // First extract all the sub strings in the mod string
        // for each channel. e.g. C+m?,1;C+h?,2; will get split
        // into 2 substrings.
        std::vector<std::pair<size_t, size_t>> delims;
        size_t pos = 0;
        while (pos < modbase_str.length()) {
            size_t delim_pos = modbase_str.find_first_of(';', pos);
            delims.push_back({pos, delim_pos});
            pos = delim_pos + 1;
        }
        size_t prob_pos = 0;  // Track which probability values will be needed.
        // Iterate over each substring, and fetch the count values.
        for (auto [a, b] : delims) {
            std::string prefix = "";
            std::string counts = "";
            int bases_seen = 0;
            bool in_counts = false;
            pos = a;
            bool found_start = false;
            while (pos < b) {
                auto comma_pos = std::min(modbase_str.find_first_of(',', pos), b);
                auto substr_len = comma_pos - pos;
                // Begining of each substring is the channel prefix. Counts only
                // start after the first comma is seen.
                if (!in_counts) {
                    in_counts = true;
                    prefix = modbase_str.substr(pos, substr_len);
                } else {
                    int num_skips = std::stoi(modbase_str.substr(pos, substr_len));
                    bases_seen += num_skips;  // Add the intervening non-modified bases.
                    if (bases_seen >= end) {
                        // Do nothing as these modbases are trimmed.
                    } else if (bases_seen >= start) {
                        // Once we reach the trim start position, the first
                        // skip count needs to be adjusted.
                        if (!found_start) {
                            counts += "," + std::to_string(bases_seen - start);
                            found_start = true;
                        } else {
                            counts += "," + std::to_string(num_skips);
                        }
                        if (!modbase_probs.empty()) {
                            trimmed_modbase_probs.push_back(modbase_probs[prob_pos]);
                        }
                    }
                    prob_pos++;
                    bases_seen++;  // Add one more to account for the actual modified base.
                }
                pos = comma_pos + 1;
            }
            if (!counts.empty()) {
                trimmed_modbase_str += prefix + counts + ";";
            }
        }
    }
    return {trimmed_modbase_str, trimmed_modbase_probs};
}

}  // namespace dorado::utils
