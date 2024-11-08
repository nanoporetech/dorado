#pragma once

// #include <iterator>
// #include <type_traits>

#include <cstdint>

namespace dorado {

template <typename T>
constexpr std::int64_t ssize(const T& obj) noexcept {
    return static_cast<int64_t>(std::size(obj));
}

// template <class Container>
// constexpr int64_t ssize(const Container& c) noexcept
//     -> std::common_type_t<std::ptrdiff_t, std::make_signed_t<decltype(c.size())>>
// {
//     return static_cast<std::common_type_t<std::ptrdiff_t, std::make_signed_t<decltype(c.size())>>>(c.size());
// }

// template <class T, std::size_t N>
// constexpr std::ptrdiff_t ssize(const T (&array)[N]) noexcept {
//     return static_cast<std::ptrdiff_t>(N);
// }

}  // namespace dorado
