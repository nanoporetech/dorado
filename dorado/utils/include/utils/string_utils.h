#pragma once

#include <algorithm>
#include <cctype>
#include <charconv>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace dorado::utils {

[[nodiscard]] inline std::vector<std::string> split(std::string_view input, char delimiter) {
    std::vector<std::string> result;
    size_t pos;
    while ((pos = input.find(delimiter)) != std::string_view::npos) {
        result.push_back(std::string(input.substr(0, pos)));
        input.remove_prefix(pos + 1);
    }

    result.push_back(std::string(input));
    return result;
}

[[nodiscard]] inline std::vector<std::string_view> split_view(const std::string_view input,
                                                              const char delimiter) {
    if (std::empty(input)) {
        return {};
    }
    size_t start = 0;
    size_t pos = 0;
    std::vector<std::string_view> result;
    while ((pos = input.find(delimiter, start)) != std::string_view::npos) {
        result.emplace_back(std::string_view(std::data(input) + start, pos - start));
        start = pos + 1;
    }
    result.emplace_back(std::string_view(std::data(input) + start, std::size(input) - start));
    return result;
}

template <typename StringLike>
[[nodiscard]] inline std::string join(const std::vector<StringLike>& inputs,
                                      std::string_view separator) {
    std::size_t total_size = inputs.empty() ? 0 : (inputs.size() - 1) * separator.size();
    for (const auto& item : inputs) {
        total_size += item.size();
    }

    std::string result;
    result.reserve(total_size);
    for (const auto& item : inputs) {
        if (!result.empty()) {
            result += separator;
        }
        result += item;
    }
    return result;
}

[[nodiscard]] inline bool starts_with(std::string_view str, std::string_view prefix) {
    return str.rfind(prefix, 0) != std::string::npos;
}

[[nodiscard]] inline bool ends_with(std::string_view str, std::string_view suffix) {
    if (str.length() < suffix.length()) {
        return false;
    }
    return str.substr(str.length() - suffix.length()) == suffix;
}

[[nodiscard]] inline bool contains(std::string_view str, std::string_view substr) {
    return str.find(substr) != str.npos;
}

inline void rtrim(std::string& s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) { return !std::isspace(ch); }).base(),
            s.end());
}

[[nodiscard]] inline std::string_view rtrim_view(const std::string& s) {
    const auto last_char_it =
            std::find_if(s.rbegin(), s.rend(), [](int ch) { return !std::isspace(ch); }).base();
    return std::string_view(s.data(), last_char_it - s.begin());
}

[[nodiscard]] inline std::string to_uppercase(std::string in) {
    std::transform(in.begin(), in.end(), in.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return in;
}

template <typename T>
[[nodiscard]] inline std::optional<T> from_chars(std::string_view str) {
    static_assert(std::is_integral_v<T>,
                  "libc++ (macOS) won't provide floating point support until they're on LLVM 20 "
                  "(_LIBCPP_VERSION >= 200000)");

    T value = 0;
    const auto res = std::from_chars(str.data(), str.data() + str.size(), value);
    if (res.ec != std::errc{}) {
        return std::nullopt;
    }
    return value;
}

namespace detail {
template <typename Int>
static constexpr std::size_t min_space_required()
    requires std::is_integral_v<Int>
{
    // 1 for sign, 1 for rounding in numeric_limits, extra for safety.
    return std::numeric_limits<Int>::digits10 + 1 + 1 + 2;
}
template <typename Float>
static constexpr std::size_t min_space_required()
    requires std::is_floating_point_v<Float>
{
    // 1 for sign, 1 for decimal point, 1 for rounding in numeric_limits, 5 for e+123, extra for safety.
    return std::numeric_limits<Float>::max_digits10 + 1 + 1 + 1 + 5 + 2;
}
}  // namespace detail

// Write the integer to the buffer, returning a span of the written string.
template <std::size_t BufferSize, typename Int>
[[nodiscard]] inline constexpr std::string_view to_chars(Int value, char* buffer)
    requires std::is_integral_v<Int>
{
    constexpr auto min_size = detail::min_space_required<Int>();
    static_assert(BufferSize >= min_size);
    const auto res = std::to_chars(buffer, buffer + BufferSize, value);
    if (res.ec != std::errc{}) [[unlikely]] {
        // This shouldn't happen unless the size calculation is wrong.
        throw std::logic_error("to_chars() failed due to lack of space");
    }
    return std::string_view{buffer, res.ptr};
}

// Write the float to the buffer, returning a span of the written string.
template <std::size_t BufferSize, typename Float>
[[nodiscard]] inline constexpr std::string_view to_chars(Float value, char* buffer)
    requires std::is_floating_point_v<Float>
{
    constexpr auto min_size = detail::min_space_required<Float>();
    static_assert(BufferSize >= min_size);
    // Note that this gives different output to std::to_string() but matches the output of
    // std::ostringstream. For example, this will format 5.5f as "5.5" (shortest representation)
    // whereas std::to_string() would output "5.500000" (full representation). We rely on this
    // behaviour in multiple places in ont_core.
    const auto res = std::to_chars(buffer, buffer + BufferSize, value, std::chars_format::general);
    if (res.ec != std::errc{}) [[unlikely]] {
        // This shouldn't happen unless the size calculation is wrong.
        throw std::logic_error("to_chars() failed due to lack of space");
    }
    return std::string_view{buffer, res.ptr};
}

// Write the value to the fixed size buffer, returning a span of the written string.
template <typename T, std::size_t BufferSize>
[[nodiscard]] inline constexpr std::string_view to_chars(T value, char (&buffer)[BufferSize]) {
    return to_chars<BufferSize>(value, buffer);
}

// Write the value to a stack buffer and return it.
template <typename T>
[[nodiscard]] inline constexpr auto to_chars(T value) {
    constexpr auto min_size = detail::min_space_required<T>();
    struct CharBuffer {
        char data[min_size]{};
        std::size_t size = 0;

        std::string_view view() const { return {data, size}; }
        explicit operator std::string_view() const { return view(); }
    };

    CharBuffer char_buffer;
    const auto view = to_chars(value, char_buffer.data);
    char_buffer.size = view.size();
    return char_buffer;
}

}  // namespace dorado::utils
