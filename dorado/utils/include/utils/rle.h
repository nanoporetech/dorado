#pragma once

#include <cstdint>
#include <functional>
#include <tuple>
#include <vector>

namespace dorado {

template <typename T, typename Compare>
std::vector<std::tuple<int64_t, int64_t, T>> run_length_encode(const std::vector<T> &data,
                                                               Compare &&comp) {
    std::vector<std::tuple<int64_t, int64_t, T>> result;

    if (std::empty(data)) {
        return result;
    }

    int64_t start = 0;

    for (int64_t i = 1; i < static_cast<int64_t>(std::size(data)); ++i) {
        if (!comp(data[i], data[start])) {
            result.emplace_back(start, i, data[start]);
            start = i;
        }
    }

    result.emplace_back(start, static_cast<int64_t>(std::size(data)), data[start]);

    return result;
}

template <typename T>
auto run_length_encode(const std::vector<T> &data) {
    return run_length_encode(data, std::equal_to<T>());
}

}  // namespace dorado
