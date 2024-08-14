#pragma once
#include "types.h"

#include <edlib.h>

#include <string>

namespace dorado::utils {

/**
 * @brief Generates a human-readable string representation of a pairwise sequence alignment.
 *
 * This function takes two C-style strings, `query` and `target`, and an `EdlibAlignResult`
 * object, `result`, and returns a formatted string representation of their alignment.
 * The `EdlibAlignResult` is a data structure from the Edlib library, which provides an
 * efficient and accurate pairwise sequence alignment algorithm.
 *
 * The function generates a formatted alignment string, which consists of:
 * 1. The aligned `query` sequence with deletions represented by dashes '-'
 * 2. A separator line with '|' for matches, '*' for mismatches, and spaces ' ' for insertions and deletions
* 3. The aligned `target` sequence with insertions represented by dashes '-'
 *
 * @param query A pointer to the query C-style string
 * @param target A pointer to the target C-style string
 * @param result A reference to an EdlibAlignResult object containing the alignment information
 * @return A formatted string representation of the alignment
 */

std::string alignment_to_str(const char* query, const char* target, const EdlibAlignResult& result);

}  // namespace dorado::utils
