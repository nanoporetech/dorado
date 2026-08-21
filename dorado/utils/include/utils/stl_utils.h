#pragma once

#include <cstdint>
#include <functional>
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

}  // namespace dorado::utils
