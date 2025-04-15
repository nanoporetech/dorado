#include "consensus_utils.h"

#include "utils/ssize.h"

#include <stdexcept>

namespace dorado::secondary {

/**
 * \brief Copy the draft sequence for a given sample, and expand it with '*' in places of gaps.
 */
std::string extract_draft_with_gaps(const std::string& draft,
                                    const std::vector<int64_t>& positions_major,
                                    const std::vector<int64_t>& positions_minor) {
    if (std::size(positions_major) != std::size(positions_minor)) {
        throw std::runtime_error(
                "The positions_major and positions_minor are not of the same size! "
                "positions_major.size = " +
                std::to_string(std::size(positions_major)) +
                ", positions_minor.size = " + std::to_string(std::size(positions_minor)));
    }

    const int64_t draft_len = dorado::ssize(draft);

    std::string ret(std::size(positions_major), '*');

    for (int64_t i = 0; i < dorado::ssize(positions_major); ++i) {
        if ((positions_major[i] < 0) || (positions_major[i] >= draft_len)) {
            throw std::runtime_error(
                    "The positions_major contains coordinates out of range for the input draft! "
                    "Requested coordinate: " +
                    std::to_string(positions_major[i]) +
                    ", draft len = " + std::to_string(draft_len));
        }
        ret[i] = (positions_minor[i] == 0) ? draft[positions_major[i]] : '*';
    }

    return ret;
}

}  // namespace dorado::secondary
