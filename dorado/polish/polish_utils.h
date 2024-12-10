#pragma once

#include "polish/interval.h"

#include <torch/torch.h>

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
void print_tensor_shape(std::ostream& os, const torch::Tensor& tensor);

/**
 * \brief Returns a string with a formatted tensor size.
 */
std::string tensor_shape_as_string(const torch::Tensor& tensor);

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

std::string fetch_seq(const std::filesystem::path& index_fn,
                      const std::string& seq_name,
                      int32_t start,
                      int32_t end);

/**
 * \brief Prints the contents of an iterable container to a stream. Useful for debug purposes.
 */
template <typename T>
void print_container(std::ostream& os, const T& data, const std::string& delimiter) {
    bool first = true;
    os << "[";
    for (const auto& val : data) {
        if (!first) {
            os << delimiter;
        }
        os << val;
        first = false;
    }
    os << "]";
}

/**
 * \brief Wrapper around print_container, but it returns a std::string formatted string instead.
 */
template <typename T>
std::string print_container_as_string(const T& data, const std::string& delimiter) {
    std::ostringstream oss;
    print_container(oss, data, delimiter);
    return oss.str();
}

/**
 * \brief Computes intervals of input objects to split the input data into. Similar to
 *          compute_partitions, but here the items can have variable size.
 * \param data Input data items to partition.
 * \param batch_size Approximate size of one output batch. The output batch size is allowed to be larger than this if the next item crosses the size boundary.
 * \param functor_data_size Functor used to determine the size of one of the data objects.
 * \returns Vector of intervals which divide the input data into batches of specified size.
 */
template <typename T, typename F>
std::vector<Interval> create_batches(const T& data,
                                     const int64_t batch_size,
                                     const F& functor_data_size) {
    std::vector<Interval> ret;
    Interval interval{0, 0};
    int64_t sum = 0;
    for (const auto& val : data) {
        const int64_t s = functor_data_size(val);
        sum += s;
        ++interval.end;
        if (sum >= batch_size) {
            ret.emplace_back(interval);
            interval.start = interval.end;
            sum = 0;
        }
    }
    if (interval.end > interval.start) {
        ret.emplace_back(interval);
    }
    return ret;
}

void save_tensor(const torch::Tensor& tensor, const std::string& file_path);

}  // namespace dorado::polisher
