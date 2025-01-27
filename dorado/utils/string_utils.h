#pragma once

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>
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

inline std::string join(const std::vector<std::string>& inputs, const std::string& separator) {
    std::string result;
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

}  // namespace dorado::utils
