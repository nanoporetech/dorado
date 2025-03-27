#include "container_utils.h"

#include <algorithm>
#include <sstream>

namespace dorado::utils {

std::vector<int32_t> parse_int32_vector(const std::string& input) {
    if (std::empty(input)) {
        return {};
    }
    if ((std::size(input) < 2) || (input.front() != '[') || (input.back() != ']')) {
        throw std::runtime_error("Input string must start with '[' and end with ']'.");
    }

    // Remove the brackets and trim the string
    std::string trimmed = input.substr(1, std::size(input) - 2);
    trimmed.erase(std::remove(std::begin(trimmed), std::end(trimmed), ' '), std::end(trimmed));
    trimmed.erase(std::remove(std::begin(trimmed), std::end(trimmed), '\t'), std::end(trimmed));

    std::vector<int32_t> result;
    std::istringstream ss(trimmed);
    std::string token;

    while (std::getline(ss, token, ',')) {
        if (std::empty(token)) {
            continue;
        }
        result.push_back(std::stoi(token));
    }

    return result;
}

}  // namespace dorado::utils
