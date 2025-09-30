#pragma once

#include <algorithm>
#include <cctype>
#include <charconv>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace dorado::utils {

inline std::vector<std::string> split(std::string_view input, char delimiter) {
    std::vector<std::string> result;
    size_t pos;
    while ((pos = input.find(delimiter)) != std::string_view::npos) {
        result.push_back(std::string(input.substr(0, pos)));
        input.remove_prefix(pos + 1);
    }

    result.push_back(std::string(input));
    return result;
}

inline std::vector<std::string_view> split_view(const std::string_view input,
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
inline std::string join(const std::vector<StringLike>& inputs, std::string_view separator) {
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

inline bool starts_with(std::string_view str, std::string_view prefix) {
    return str.rfind(prefix, 0) != std::string::npos;
}

inline bool ends_with(std::string_view str, std::string_view suffix) {
    if (str.length() < suffix.length()) {
        return false;
    }
    return str.substr(str.length() - suffix.length()) == suffix;
}

inline bool contains(std::string_view str, std::string_view substr) {
    return str.find(substr) != str.npos;
}

inline void rtrim(std::string& s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) { return !std::isspace(ch); }).base(),
            s.end());
}

inline std::string_view rtrim_view(const std::string& s) {
    const auto last_char_it =
            std::find_if(s.rbegin(), s.rend(), [](int ch) { return !std::isspace(ch); }).base();
    return std::string_view(s.data(), last_char_it - s.begin());
}

inline std::string to_uppercase(std::string in) {
    std::transform(in.begin(), in.end(), in.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return in;
}

template <typename T>
inline std::optional<T> from_chars(std::string_view str) {
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

}  // namespace dorado::utils
