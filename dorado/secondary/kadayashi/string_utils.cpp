#include "string_utils.h"

#include <cstring>
#include <sstream>
#include <vector>

namespace kadayashi {

bool stoi_catch(int &dest, const char *s) {
    try {
        size_t tmp = 0;
        dest = std::stoi(s, &tmp);
        if (tmp == 0 || tmp != strlen(s)) {
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

bool stod_catch(double &dest, const char *s) {
    try {
        size_t tmp;
        dest = std::stod(s, &tmp);
        if (tmp != strlen(s)) {
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

std::string create_region_string(const std::string_view ref_name,
                                 const uint32_t start,
                                 const uint32_t end) {
    std::string ret(ref_name);
    ret.append(":");
    ret.append(std::to_string(start + 1));
    ret.append("-");
    ret.append(std::to_string(end));
    return ret;
}

bool split_region_string(const std::string &itvl_str,  // 1-index []
                         std::string &tname,
                         uint32_t &start,
                         uint32_t &end  // 0-index [)
) {
    std::vector<std::string> fields;
    std::istringstream ss(itvl_str);
    std::string field;
    while (std::getline(ss, field, ':')) {
        fields.push_back(field);
    }
    if (fields.size() != 2) {
        return false;
    }

    std::istringstream ss2(fields.back());
    fields.pop_back();
    while (std::getline(ss2, field, '-')) {
        fields.push_back(field);
    }
    if (fields.size() != 3) {
        return false;
    }

    int si = 0;
    int ei = 0;
    const bool ok1 = stoi_catch(si, fields[1].c_str());
    const bool ok2 = stoi_catch(ei, fields[2].c_str());
    if (!ok1 || !ok2 || si <= 0 || ei <= 0 || si > ei || fields[0].empty()) {
        return false;
    } else {
        tname = fields[0];
        start = static_cast<uint32_t>(si - 1);
        end = static_cast<uint32_t>(ei);
        return true;
    }
}

}  // namespace kadayashi
