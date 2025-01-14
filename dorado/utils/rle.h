#pragma once

#include <cstdint>
#include <functional>
#include <tuple>
#include <vector>

namespace dorado {

template <typename T, typename Compare = std::function<bool(const T&, const T&)>>
std::vector<std::tuple<int64_t, int64_t, T>> run_length_encode(const std::vector<T>& data,
                                                               Compare comp = std::equal_to<T>()) {
    std::vector<std::tuple<int64_t, int64_t, T>> result;

    if (std::empty(data)) {
        return result;
    }

    int64_t start = 0;
    T current_value = data[0];

    for (int64_t i = 1; i < static_cast<int64_t>(std::size(data)); ++i) {
        if (!comp(data[i], current_value)) {
            result.emplace_back(start, i, current_value);
            start = i;
            current_value = data[i];
        }
    }

    result.emplace_back(start, static_cast<int64_t>(std::size(data)), current_value);

    return result;
}

}  // namespace dorado
