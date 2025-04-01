#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dorado::secondary {

std::string extract_draft_with_gaps(const std::string& draft,
                                    const std::vector<int64_t>& positions_major,
                                    const std::vector<int64_t>& positions_minor);

}  // namespace dorado::secondary
