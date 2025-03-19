#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dorado::alignment {

/// <summary>
/// Collection of sequence record name/length pairs
/// </summary>
using HeaderSequenceRecords = std::vector<std::pair<std::string, uint32_t>>;

/// <summary>
/// Possible results when loading an index file.
/// </summary>
enum class IndexLoadResult {
    reference_file_not_found,
    validation_error,
    no_index_loaded,
    end_of_index,
    file_open_error,
    success,
};

}  // namespace dorado::alignment
