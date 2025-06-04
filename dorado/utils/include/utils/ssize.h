#pragma once

#include <cstdint>

namespace dorado {

template <typename T>
constexpr std::int64_t ssize(const T& obj) noexcept {
    return static_cast<int64_t>(std::size(obj));
}

}  // namespace dorado
