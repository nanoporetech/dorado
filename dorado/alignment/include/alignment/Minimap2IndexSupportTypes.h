#pragma once

namespace dorado::alignment {

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
