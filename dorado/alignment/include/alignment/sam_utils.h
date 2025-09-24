#include "utils/types.h"

#include <string>
#include <string_view>
#include <vector>

namespace dorado::alignment {

int parse_cigar(std::string_view cigar, dorado::AlignmentResult& result);

std::vector<AlignmentResult> parse_sam_lines(const std::string& sam_content,
                                             const std::string& query_seq,
                                             const std::string& query_qual);

}  // namespace dorado::alignment