#pragma once

#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <vector>

namespace dorado::utils {

template <typename T>
T div_round_closest(const T n, const T d) {
    return ((n < 0) ^ (d < 0)) ? ((n - d / 2) / d) : ((n + d / 2) / d);
}
template <typename T>
T div_round_up(const T a, const T b) {
    return (a + b - 1) / b;
}
template <typename T>
T pad_to(const T a, const T b) {
    return div_round_up(a, b) * b;
}

// Adapted from https://stackoverflow.com/questions/11964552/finding-quartiles
template <typename T, typename = typename std::enable_if<std::is_floating_point<T>::value, T>::type>
inline std::vector<T> quantiles(const std::vector<T>& in_data, const std::vector<T>& quants) {
    NVTX3_FUNC_RANGE();
    if (in_data.empty()) {
        return {};
    }

    if (in_data.size() == 1) {
        return {in_data.front()};
    }

    auto data = in_data;
    std::sort(std::begin(data), std::end(data));
    std::vector<T> quantiles;
    quantiles.reserve(quants.size());

    auto linear_interp = [](T v0, T v1, T t) { return (1 - t) * v0 + t * v1; };

    for (size_t i = 0; i < quants.size(); ++i) {
        T pos = linear_interp(0, T(data.size() - 1), quants[i]);

        int64_t left = std::max(int64_t(std::floor(pos)), int64_t(0));
        int64_t right = std::min(int64_t(std::ceil(pos)), int64_t(data.size() - 1));
        T data_left = data.at(left);
        T data_right = data.at(right);

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
    NVTX3_FUNC_RANGE();
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

template <typename T>
bool eq_with_tolerance(T a, T b, T tol) {
    return std::abs(a - b) <= tol;
}

}  // namespace dorado::utils
