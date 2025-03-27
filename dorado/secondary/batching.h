#pragma once

#include "interval.h"
#include "region.h"
#include "utils/ssize.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace dorado::secondary {

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

/**
 * \brief Groups input regions into bins of sequence IDs and batches of approximate specified size.
 * \param draft_lens Vector of sequence names and their lengths.
 * \param user_regions Vector of user-provided regions. Optional, can be empty.
 * \param draft_batch_size Split draft regions into batches of roughly this size. Each element is an interval
 *                          of draft sequences.
 * \returns Pair of two objects: (1) vector of sequence IDs (0 to len(draft_lens)), with the internal vector containing
 *          regions for that sequence; (2) vector of intervals of roughly batch_size bases (or more if a sequence is larger).
 *          These intervals are indices of the first return vector in this pair.
 */
std::pair<std::vector<std::vector<Region>>, std::vector<Interval>> prepare_region_batches(
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        const std::vector<Region>& user_regions,
        const int64_t draft_batch_size);

}  // namespace dorado::secondary
