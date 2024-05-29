#include "myers.h"

#include "utils/PostCondition.h"
#include "utils/alignment_utils.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iomanip>
#include <optional>
#include <ostream>

namespace dorado::splitter {

namespace {

// Returns the array D[0..n], where D[i] is equal to local alignment of the pattern to suffix of text[0..i).
std::vector<size_t> d_myers(const char* pattern, size_t m, const char* text, size_t n) {
    constexpr size_t MAX_ALPHABET = 256;
    assert(m < 64);
    uint64_t PM[MAX_ALPHABET]{};
    for (size_t i = 0; i < m; i++) {
        PM[static_cast<uint8_t>(pattern[i])] |= uint64_t{1} << i;
    }

    std::vector<size_t> D(n + 1);
    uint64_t VP = ~uint64_t{0};
    uint64_t VN = 0;
    size_t score = m;
    D[0] = score;
    for (size_t j = 0; j < n; j++) {
        const uint64_t EQ = PM[static_cast<uint8_t>(text[j])];
        const uint64_t D0 = (((EQ & VP) + VP) ^ VP) | EQ | VN;
        uint64_t HP = VN | ~(D0 | VP);
        uint64_t HN = D0 & VP;

        if (HP & (uint64_t{1} << (m - 1))) {
            score++;
        }
        if (HN & (uint64_t{1} << (m - 1))) {
            score--;
        }
        D[j + 1] = score;

        HP <<= 1;
        HN <<= 1;
        VP = HN | ~(D0 | HP);
        VN = D0 & HP;
    }
    return D;
}

}  // namespace

std::vector<EdistResult> myers_align(std::string_view query,
                                     std::string_view seq,
                                     std::size_t max_edist) {
    std::vector<EdistResult> ranges;
    const auto query_len = query.size();
    if (seq.size() < query_len) {
        // Too small, don't bother.
        return ranges;
    }

    auto add_match = [&](std::size_t end, std::size_t edist) {
        // |edist| is for the full query ending at |end|, so we know the earliest that the match can start.
        const auto max_match_len = std::min(query_len + edist, end);
        const auto min_match_start = end - max_match_len;

        if (edist == 0) {
            // Exact match, nothing more to do.
            ranges.push_back({min_match_start, end, edist});

        } else {
            // If this isn't an exact match then hand over to edlib to get the start index.
            const auto max_match_span = seq.substr(min_match_start, max_match_len);
            auto edlib_cfg = edlibNewAlignConfig(static_cast<int>(edist), EDLIB_MODE_HW,
                                                 EDLIB_TASK_LOC, nullptr, 0);
            auto edlib_result =
                    edlibAlign(query.data(), static_cast<int>(query.size()), max_match_span.data(),
                               static_cast<int>(max_match_span.size()), edlib_cfg);
            assert(edlib_result.status == EDLIB_STATUS_OK);
            auto edlib_cleanup = utils::PostCondition([&] { edlibFreeAlignResult(edlib_result); });

            if (edlib_result.status == EDLIB_STATUS_OK) {
                // edlib only reports the best edist it finds, so if there's a better match in the same span then it'll
                // report that instead. This can happen for spans that are close together, for example:
                // edists:...,7,6,5,5,4,5,6,6,6,5,6,..., max_edist=5
                //               end1=^    end2=^
                // When processing 'end2' we have to extend our search span to include that of 'end1' (since it's
                // |max_edist| indices away). This means that, if the edits aren't insertions, we end up finding the
                // same sequence as |end1|, which has a better edist and hence is the only thing edlib reports.
                // We should be safe to ignore the worse edist in those cases.
                for (int i = 0; i < edlib_result.numLocations; i++) {
                    // edlib indices are inclusive, ours are exclusive.
                    const std::size_t edlib_end =
                            min_match_start + edlib_result.endLocations[i] + 1;
                    if (edlib_result.editDistance == static_cast<int>(edist) && edlib_end == end) {
                        ranges.push_back(
                                {min_match_start + edlib_result.startLocations[i], end, edist});
                    }
                }
            }
        }
    };

    // Calculate edit distances for each index.
    const auto local_edists = d_myers(query.data(), query.size(), seq.data(), seq.size());

    // Look for drops below the threshold and join neighbouring ranges together.
    //
    // TODO: can we improve on this, since we have cases such as:
    // local_edists = ...,7,6,5,6,6,5,5,6,5,4,5,5,6,7,..., max_edist = 5
    // hits:                 [^]   [^  ] [  ^   ]
    // where these hits should probably be grouped together.
    //
    std::size_t current_range_end = 0;
    bool in_range = false;
    std::size_t min_edist = max_edist + 1;
    for (std::size_t current_end = query_len; current_end < local_edists.size(); current_end++) {
        // See if this is a better match.
        const auto current_edist = local_edists[current_end];
        if (current_edist <= max_edist && current_edist < min_edist) {
            min_edist = current_edist;
            current_range_end = current_end;
            in_range = true;
        } else if (in_range && current_edist > max_edist) {
            // We're back out of the threshold, so end the current range.
            add_match(current_range_end, min_edist);
            in_range = false;
            min_edist = max_edist + 1;
        }
    }

    // End the current range if we're in one.
    if (in_range) {
        add_match(current_range_end, min_edist);
    }

    // Deduplicate the ranges.
    auto less = [](const EdistResult& lhs, const EdistResult& rhs) { return lhs.end < rhs.end; };
    auto equal = [](const EdistResult& lhs, const EdistResult& rhs) {
        return lhs.begin == rhs.begin && lhs.end == rhs.end;
    };
    std::sort(ranges.begin(), ranges.end(), less);
    auto new_end = std::unique(ranges.begin(), ranges.end(), equal);
    ranges.erase(new_end, ranges.end());

    return ranges;
}

void print_edists(std::ostream& os, std::string_view seq, const std::vector<size_t>& edists) {
    assert(edists.size() == seq.size() + 1);

    // Print the sequence with spaces to match the edists.
    for (char c : seq) {
        os << "  " << c;
    }
    os << '\n';

    // Print the edists. Maximum edist is 64, so we only need to pad 2 spaces.
    const auto old_fill = os.fill(' ');
    for (size_t s : edists) {
        os << std::setw(3) << s;
    }
    os.fill(old_fill);
    os << '\n';
}

}  // namespace dorado::splitter
