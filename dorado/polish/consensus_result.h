#pragma once

#include <cstdint>
#include <string>

namespace dorado::polisher {

struct ConsensusResult {
    std::string seq;
    std::string quals;
    int32_t draft_id = -1;    // Draft sequence ID where this comes from.
    int64_t draft_start = 0;  // Position in draft where seq begins.
    int64_t draft_end = 0;    // Position in draft where seq ends.
};

}  // namespace dorado::polisher