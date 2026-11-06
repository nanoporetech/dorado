#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <tuple>
#include <vector>

namespace dorado::utils {

// `Map::mapped_type` is a non-deduced context, so default_value (e.g. a `const char*`)
// converts to it via normal overload resolution instead of forcing a mismatched Value.
template <typename Map, typename Key>
typename Map::mapped_type get_value_or_default(const Map& map,
                                               const Key& key,
                                               const typename Map::mapped_type& default_value) {
    const auto it = map.find(key);
    return it != std::cend(map) ? it->second : default_value;
}

/**
 * \brief Returns the first column at or after the given reference query.
 * \param values Vector of values to search. It needs to be sorted.
 * \param query Query value to search.
 * \returns Integer position of the found value or the first larger one. Edge cases:
 *          1. If the query is larger than the last element, it returns the size
 *              of the input vector.
 *          2. If the value is smaller than the first element, it returns 0.
 *          3. If the value cannot be found, it returns the first larger element.
 * \details This performs a binary search internally, so make sure that the
 *          data is sorted.
 * \param
 */
template <typename T>
int64_t find_column_to_insert(const std::vector<T>& values, const T query) {
    return std::distance(std::begin(values),
                         std::lower_bound(std::begin(values), std::end(values), query));
}

}  // namespace dorado::utils
