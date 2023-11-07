#pragma once

#include <cstdint>
#include <vector>

namespace dorado::alignment {

/// <summary>
/// Collection of sequence record name/length pairs
/// </summary>
using HeaderSequenceRecords = std::vector<std::pair<char *, uint32_t>>;

/// <summary>
/// Possible results when loading an index file.
/// </summary>
enum class IndexLoadResult {
    reference_file_not_found,
    split_index_not_supported,
    validation_error,
    success,
};

}  // namespace dorado::alignment