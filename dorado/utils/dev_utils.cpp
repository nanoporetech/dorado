#include "dev_utils.h"

namespace dorado::utils::details {

std::unordered_map<std::string, std::pair<double, bool>> g_dev_options;

void extract_dev_options(const std::string& env_string) {
    constexpr char SEPARATOR = ';';
    std::vector<std::string> parts;
    size_t start = 0;
    for (size_t end = env_string.find(SEPARATOR, start); end != std::string::npos;) {
        parts.emplace_back(env_string, start, end - start);
        start = end + 1;
        end = env_string.find(SEPARATOR, start);
    }
    parts.emplace_back(env_string, start, std::string::npos);

    for (auto& part : parts) {
        double value = 1;
        size_t eq_pos = part.find('=');
        if (eq_pos != std::string::npos) {
            value = strtod(part.c_str() + eq_pos + 1, nullptr);
        }
        g_dev_options[part.substr(0, eq_pos)] = {value, true};
    }
}

}  // namespace dorado::utils::details
