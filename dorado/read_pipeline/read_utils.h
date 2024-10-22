#pragma once

#include "messages.h"

namespace dorado::utils {
SimplexReadPtr shallow_copy_read(const SimplexRead& read);

// Find the trimming index for degraded ends of a mux_change read.
int64_t find_mux_change_trim_seq_index(const std::string& qstring);

// Given a read, only trims reads which have end_reason `mux_change` or `unblock_mux_change` as
// the end of the sequence has been degraded.
void mux_change_trim_read(ReadCommon& read_common);

}  // namespace dorado::utils
