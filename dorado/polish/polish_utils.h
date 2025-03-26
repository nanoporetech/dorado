#pragma once

#include "interval.h"
#include "utils/region.h"

#include <ATen/ATen.h>

#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <sstream>
#include <string>
#include <vector>

namespace dorado::polisher {

/**
 * \brief Prints the tensor size to a stream.
 */
void print_tensor_shape(std::ostream& os, const at::Tensor& tensor, const std::string& delimiter);

/**
 * \brief Returns a string with a formatted tensor size.
 */
std::string tensor_shape_as_string(const at::Tensor& tensor);

/**
 * \brief Parses a string of form "[1, 17]" into a std::vector.
 */
std::vector<int32_t> parse_int32_vector(const std::string& input);

/**
 * \brief Utility function to determine partitions for multithreading. For example,
 *          num_items is the number of items to process, while num_chunks is the number
 *          of threads. The items are then chunked into as equal number of bins as possible.
 * \param num_items Number of items to process.
 * \param num_partitions Number of threads/buckets/partitions to divide the items into.
 * \returns Vector of intervals which is the length of min(num_items, num_chunks). Each partition
 *          is given with a start (zero-based) and end (non-inclusive) item ID.
 */
std::vector<Interval> compute_partitions(const int32_t num_items, const int32_t num_partitions);

void save_tensor(const at::Tensor& tensor, const std::string& file_path);

}  // namespace dorado::polisher
