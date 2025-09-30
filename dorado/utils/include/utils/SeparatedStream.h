#pragma once

#include "string_utils.h"

namespace dorado::utils {

/**
 * Stream-like object that splits elements on @a Separator.
 * Note that trailing elements are ignored.
 */
template <char Separator>
class SeparatedStream {
    std::string_view m_data;
    bool m_eof = false;

    template <typename T>
    static void parse_value(std::string_view in, T& out) requires std::is_integral_v<T> {
        auto parsed = utils::from_chars<T>(in);
        if (parsed.has_value()) {
            out = parsed.value();
        } else {
            out = 0;
        }
    }

    template <typename T>
    static void parse_value(std::string_view in, T& out)
            requires(std::is_same_v<T, std::string_view> || std::is_same_v<T, std::string>) {
        out = in;
    }

public:
    explicit SeparatedStream(std::string_view data) : m_data(data) {}

    // See if this stream has set the EOF bit.
    // In line with the stdlib streams, this is only set once an
    // attempt is made to read past the end.
    [[nodiscard]] bool eof() const { return m_eof; }
    [[nodiscard]] operator bool() const { return !eof(); }

    // Peek at the next element in the stream.
    [[nodiscard]] std::optional<std::string_view> peek() const {
        if (m_data.empty()) {
            return std::nullopt;
        }
        auto end = m_data.find(Separator);
        return m_data.substr(0, end);
    }

    // Read up to the next separator, same as std::getline().
    [[nodiscard]] std::optional<std::string_view> getline() {
        const auto next = peek();
        if (!next.has_value()) {
            m_eof = true;
            return std::nullopt;
        }
        const auto elem = next.value();

        // Consume the element and the separator.
        m_data.remove_prefix(elem.size());
        if (!m_data.empty()) {
            m_data.remove_prefix(1);
        }
        return elem;
    }

    template <typename T>
    SeparatedStream& operator>>(T& val) {
        auto column_data = getline();
        if (column_data.has_value()) {
            parse_value<T>(column_data.value(), val);
        } else {
            val = {};
        }
        return *this;
    }
};

using NewlineSeparatedStream = SeparatedStream<'\n'>;
using SpaceSeparatedStream = SeparatedStream<' '>;
using TabSeparatedStream = SeparatedStream<'\t'>;

}  // namespace dorado::utils
