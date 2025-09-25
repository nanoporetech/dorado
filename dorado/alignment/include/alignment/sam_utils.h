#include "utils/types.h"

#include <string_view>
#include <vector>

namespace dorado::alignment {

int parse_cigar(std::string_view cigar, dorado::AlignmentResult& result);

std::vector<AlignmentResult> parse_sam_lines(std::string_view sam_content,
                                             std::string_view query_seq,
                                             std::string_view query_qual);

}  // namespace dorado::alignment
