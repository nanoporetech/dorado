#include "secondary/common/variant.h"

#include "utils/container_utils.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <ostream>
#include <string_view>
#include <tuple>

namespace dorado::secondary {

std::ostream& operator<<(std::ostream& os, const Variant& v) {
    const auto print_map = [&os](const std::string_view name, const auto& paired_data) {
        os << name << ':';
        bool first = true;
        for (const auto& [key, val] : paired_data) {
            if (!first) {
                os << ',';
            }
            os << key << '=' << val;
            first = false;
        }
    };

    os << v.seq_id << '\t' << v.pos << '\t' << v.ref << "\t{";
    utils::print_container(os, v.alts, ",", true);
    os << "}\t" << v.filter << '\t' << v.qual << '\t' << v.rstart << '\t' << v.rend;
    os << '\t';
    print_map("gt", v.format);
    os << '\t';
    print_map("info", v.info);

    return os;
}

std::string to_string(const Variant& v) {
    std::ostringstream oss;
    oss << v;
    return oss.str();
}

bool operator==(const Variant& lhs, const Variant& rhs) {
    return std::tie(lhs.seq_id, lhs.pos, lhs.ref, lhs.alts, lhs.filter, lhs.info, lhs.qual,
                    lhs.format, lhs.rstart,
                    lhs.rend) == std::tie(rhs.seq_id, rhs.pos, rhs.ref, rhs.alts, rhs.filter,
                                          rhs.info, rhs.qual, rhs.format, rhs.rstart, rhs.rend);
}

bool operator<(const Variant& lhs, const Variant& rhs) {
    return std::tie(lhs.seq_id, lhs.pos, lhs.ref, lhs.alts, lhs.qual) <
           std::tie(rhs.seq_id, rhs.pos, rhs.ref, rhs.alts, rhs.qual);
}

bool is_valid(const Variant& var) {
    if (std::empty(var.ref)) {
        spdlog::trace("Variant with empty ref is invalid: {}", to_string(var));
        return false;
    }
    if (std::empty(var.alts)) {
        spdlog::trace("Variant with no alts is invalid: {}", to_string(var));
        return false;
    }
    if (std::any_of(std::cbegin(var.alts), std::cend(var.alts),
                    [&var](const std::string_view val) { return val == var.ref; })) {
        spdlog::trace("Variant with alt matching ref is invalid: {}", to_string(var));
        return false;
    }
    if (std::any_of(std::cbegin(var.alts), std::cend(var.alts),
                    [](const std::string_view val) { return std::empty(val); })) {
        spdlog::trace("Variant with empty alt string is invalid: {}", to_string(var));
        return false;
    }
    return true;
}

bool is_reference_record(const Variant& var) {
    return (var.filter == NO_VARIANT_FILTER) || std::empty(var.alts) ||
           ((std::size(var.alts) == 1) &&
            ((var.alts.front() == ".") || (var.alts.front() == "<*>")));
}

int64_t variant_end(const Variant& var) {
    if (const auto it = var.info.find("END"); it != std::cend(var.info)) {
        return static_cast<int64_t>(std::stoll(it->second));
    }
    if (const auto it = var.format.find("LEN"); it != var.format.cend()) {
        return var.pos + static_cast<int64_t>(std::stoll(it->second));
    }
    return var.pos + std::max<int64_t>(1, std::ssize(var.ref));
}

bool variant_ends_before_position(const Variant& var, const int32_t seq_id, const int64_t pos) {
    return (var.seq_id == seq_id) && (variant_end(var) <= pos);
}

bool variant_covers_position(const Variant& var, const int32_t seq_id, const int64_t pos) {
    return (var.seq_id == seq_id) && (var.pos <= pos) &&
           !variant_ends_before_position(var, seq_id, pos);
}

void set_gvcf_reference_block_end(Variant& var, const int64_t end) {
    var.alts = {"<*>"};
    var.info["END"] = std::to_string(end);
    var.format["LEN"] = std::to_string(end - var.pos);
}

float get_gvcf_reference_record_gq_margin(
        const float gq,
        const std::span<const std::pair<int64_t, float>> margins) {
    if (std::empty(margins)) {
        return 0.0f;
    }
    const auto it = std::upper_bound(std::cbegin(margins), std::cend(margins), gq,
                                     [](const float val, const auto& margin) {
                                         return val < static_cast<float>(margin.first);
                                     });
    if (it == std::cbegin(margins)) {
        return it->second;
    }
    return std::prev(it)->second;
}

}  // namespace dorado::secondary
