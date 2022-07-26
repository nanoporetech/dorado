#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <vector>

namespace utils {

inline int div_round_closest(const int n, const int d) {
    return ((n < 0) ^ (d < 0)) ? ((n - d / 2) / d) : ((n + d / 2) / d);
}

// Adapted from https://stackoverflow.com/questions/11964552/finding-quartiles
template <typename T, typename = typename std::enable_if<std::is_floating_point<T>::value, T>::type>
inline std::vector<T> quantiles(std::vector<T> data, const std::vector<T>& quants) {
    if (data.empty()) {
        return {};
    }

    if (data.size() == 1) {
        return {data.front()};
    }

    std::vector<T> quantiles;
    auto linear_interp = [](T v0, T v1, T t) { return (1 - t) * v0 + t * v1; };

    // Track the rightmost point we've sorted at as nth_element ensures everything left of that is
    // less than the nth_element, so we can reduce our range each time
    auto start_it = std::begin(data);
    for (size_t i = 0; i < quants.size(); ++i) {
        T pos = linear_interp(0, T(data.size() - 1), quants[i]);

        int64_t left = std::max(int64_t(std::floor(pos)), int64_t(0));
        int64_t right = std::min(int64_t(std::ceil(pos)), int64_t(data.size() - 1));

        // Two `nth_element` calls per quantile at O(N) is better than
        // sorting `data` at O(NlogN) for data.size >> quants.size(),
        // which is almost always going to be true
        auto nth_it = std::next(std::begin(data), left);
        std::nth_element(start_it, nth_it, std::end(data));
        T data_left = *nth_it;

        nth_it = std::next(std::begin(data), right);
        std::nth_element(start_it, nth_it, std::end(data));
        T data_right = *nth_it;
        start_it = std::next(nth_it);

        T quantile = linear_interp(data_left, data_right, pos - left);
        quantiles.push_back(quantile);
    }

    return quantiles;
}

// Perform a least-squares linear regression of the form y = mx + b, solving for m and b.
// Returns a tuple {m, b, r} where r is the regression correlation coefficient
// Adapted from https://stackoverflow.com/questions/5083465/fast-efficient-least-squares-fit-algorithm-in-c
template <typename T, typename = typename std::enable_if<std::is_floating_point<T>::value, T>::type>
std::tuple<T, T, T> linear_regression(const std::vector<T>& x, const std::vector<T>& y) {
    assert(x.size() == y.size());
    auto sum_square = [](auto s2, auto q) { return s2 + q * q; };

    T sumx2 = std::accumulate(std::begin(x), std::end(x), T(0), sum_square);
    T sumy2 = std::accumulate(std::begin(y), std::end(y), T(0), sum_square);
    T sumx = std::accumulate(std::begin(x), std::end(x), T(0));
    T sumy = std::accumulate(std::begin(y), std::end(y), T(0));

    T sumxy = 0.0;
    size_t n = x.size();
    for (size_t i = 0; i < n; ++i) {
        sumxy += x[i] * y[i];
    }

    T denom = (n * sumx2 - (sumx * sumx));
    if (denom == 0) {
        // singular matrix. can't solve the problem, return identity transform
        return {T(1), T(0), T(0)};
    }

    T m = (n * sumxy - sumx * sumy) / denom;
    T b = (sumy * sumx2 - sumx * sumxy) / denom;
    // compute correlation coeff
    T r = (sumxy - sumx * sumy / n) /
          std::sqrt((sumx2 - (sumx * sumx) / n) * (sumy2 - (sumy * sumy) / n));

    return {m, b, r};
}

}  // namespace utils