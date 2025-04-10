#include "container_utils.h"

#include <algorithm>
#include <sstream>
#include <unordered_set>

namespace dorado::utils {

std::vector<int32_t> parse_int32_vector(const std::string& input, const char delimiter) {
    if (std::empty(input)) {
        return {};
    }

    const std::unordered_set<char> open_set{'(', '[', '{', '<'};
    const std::unordered_set<char> closed_set{')', ']', '}', '>'};

    std::string trimmed;

    if ((std::size(input) >= 2) && (open_set.find(input.front()) != open_set.cend()) &&
        (closed_set.find(input.back()) != closed_set.cend())) {
        // Remove the brackets and trim the string
        trimmed = input.substr(1, std::size(input) - 2);
    } else {
        trimmed = input;
    }

    trimmed.erase(std::remove(std::begin(trimmed), std::end(trimmed), ' '), std::end(trimmed));
    trimmed.erase(std::remove(std::begin(trimmed), std::end(trimmed), '\t'), std::end(trimmed));

    std::vector<int32_t> result;
    std::istringstream ss(trimmed);
    std::string token;

    while (std::getline(ss, token, delimiter)) {
        if (std::empty(token)) {
            continue;
        }
        result.push_back(std::stoi(token));
    }

    return result;
}

}  // namespace dorado::utils
