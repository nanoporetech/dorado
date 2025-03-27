#pragma once

#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace dorado::utils {

/**
 * \brief Prints the contents of an iterable container to a stream. Useful for debug purposes.
 */
template <typename T>
void print_container(std::ostream& os, const T& data, const std::string& delimiter) {
    bool first = true;
    os << "[";
    for (const auto& val : data) {
        if (!first) {
            os << delimiter;
        }
        os << val;
        first = false;
    }
    os << "]";
}

/**
 * \brief Wrapper around print_container, but it returns a std::string formatted string instead.
 */
template <typename T>
std::string print_container_as_string(const T& data, const std::string& delimiter) {
    std::ostringstream oss;
    print_container(oss, data, delimiter);
    return oss.str();
}

/**
 * \brief Parses a string of form "[1, 17]" into a std::vector.
 * \param input Input string needs to begin with '[' and end
 *          with ']', values are comma separated, whitespaces are allowed.
 * \returns Vector of parsed integers.
 *
 * Can throw.
 */
std::vector<int32_t> parse_int32_vector(const std::string& input);

}  // namespace dorado::utils
