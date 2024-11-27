#include "trim.h"

#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <sstream>

namespace dorado::utils {

int trim(const at::Tensor& signal, float threshold, int window_size, int min_elements) {
    const int min_trim = 10;
    const int num_samples = static_cast<int>(signal.size(0)) - min_trim;
    const int num_windows = num_samples / window_size;

    // Access via raw pointers because of torch indexing overhead.
    const auto signal_f32 = signal.to(at::ScalarType::Float);
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
    if (trim_interval.first >= int(seq.length()) || trim_interval.second > int(seq.length()) ||
        trim_interval.second < trim_interval.first) {
        throw std::invalid_argument("Trim interval " + std::to_string(trim_interval.first) + "-" +
                                    std::to_string(trim_interval.second) +
                                    " is invalid for sequence " + seq);
    }
    return std::string(seq.begin() + trim_interval.first, seq.begin() + trim_interval.second);
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
        for (int i = 0; i < int(move_vals.size()); i++) {
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

std::tuple<std::string, std::vector<uint8_t>> trim_modbase_info(
        const std::string& seq,
        const std::string& modbase_str,
        const std::vector<uint8_t>& modbase_probs,
        const std::pair<int, int>& trim_interval) {
    // The algorithm below will follow an example. Say
    // seq = AATCGGAC
    // modstr = A+a?,1,0;C+m.,0;
    // probs = [10, 20, 30]
    // Trim interval = {1, 6}
    int start = trim_interval.first;
    int end = trim_interval.second;

    // Get cardinal base counts till trim start and end positions.
    std::unordered_map<char, int> bases_skipped_at_start;
    for (int i = 0; i < start; i++) {
        bases_skipped_at_start[seq[i]]++;
    }
    std::unordered_map<char, int> bases_skipped_at_end;
    for (int i = 0; i < end; i++) {
        bases_skipped_at_end[seq[i]]++;
    }

    // After counts are generated, the dicts will contain
    // bases_skipped_at_start = {A:1, T:0, C:0, G:0}
    // bases_skipped_at_end = {A:2, T:1, C:1, G:2}

    std::stringstream trimmed_modbase_str;
    std::vector<uint8_t> trimmed_modbase_probs;
    if (!modbase_str.empty()) {
        // First extract all the sub strings in the mod string
        // for each channel. e.g. A+a?,1,0;C+m.,0; will get split
        // into 2 substrings A+a?,1,0 and C+m.,0 .
        std::vector<std::string_view> mods;
        std::string_view modbase_str_view = modbase_str;
        while (!modbase_str_view.empty()) {
            size_t delim_pos = modbase_str_view.find_first_of(';');
            mods.push_back(modbase_str_view.substr(0, delim_pos));
            modbase_str_view.remove_prefix(delim_pos + 1);
        }
        // Iterate over each substring, and fetch the count values per cardinal base.
        int prob_pos = 0;  // Track the probability values to keep from the original vector.
        for (auto mod : mods) {
            std::string_view prefix;
            std::stringstream counts;
            int cardinal_bases_seen = 0;
            bool in_counts = false;
            bool found_start = false;
            int cardinal_count_at_start = 0, cardinal_count_at_end = 0;
            while (!mod.empty()) {
                auto comma_pos = mod.find_first_of(',');
                if (comma_pos == std::string::npos) {
                    comma_pos = mod.length();
                }
                // Beginning of each substring is the channel prefix. Counts only
                // start after the first comma is seen.
                if (!in_counts) {
                    in_counts = true;
                    prefix = mod.substr(0, comma_pos);  // prefixes will be A+a? and C+m.
                    char cardinal_base = prefix[0];
                    cardinal_count_at_start = bases_skipped_at_start[cardinal_base];
                    cardinal_count_at_end = bases_skipped_at_end[cardinal_base];
                } else {
                    int num_skips = std::stoi(std::string(mod.substr(0, comma_pos)));
                    cardinal_bases_seen += num_skips;  // Add the intervening non-modified bases.
                    if (cardinal_bases_seen >= cardinal_count_at_end) {
                        // Do nothing as these modbases are trimmed.
                    } else if (cardinal_bases_seen >= cardinal_count_at_start) {
                        if (!found_start) {
                            // Once we skip the number or cardinal bases till the trim start position,
                            // the skip count for the next methylated base needs to be adjusted.
                            counts << "," << (cardinal_bases_seen - cardinal_count_at_start);
                            found_start = true;
                        } else {
                            counts << "," << num_skips;
                        }
                        if (!modbase_probs.empty()) {
                            trimmed_modbase_probs.push_back(modbase_probs[prob_pos]);
                        }
                    }
                    prob_pos++;
                    cardinal_bases_seen++;  // Add one more to account for the actual modified base.
                }
                mod.remove_prefix(std::min(comma_pos + 1, mod.length()));  // No comma at the end.
            }
            trimmed_modbase_str << std::string(prefix) << counts.str() << ";";
        }
    }
    return {trimmed_modbase_str.str(), trimmed_modbase_probs};
}

}  // namespace dorado::utils
