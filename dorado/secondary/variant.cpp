#include "variant.h"

#include "utils/container_utils.h"

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
    utils::print_container(os, v.alts, ",");
    os << "}\t" << v.filter << '\t' << v.qual << '\t' << v.rstart << '\t' << v.rend;
    os << '\t';
    print_map("gt", v.genotype);
    os << '\t';
    print_map("info", v.info);

    return os;
}

bool operator==(const Variant& lhs, const Variant& rhs) {
    return std::tie(lhs.seq_id, lhs.pos, lhs.ref, lhs.alts, lhs.filter, lhs.info, lhs.qual,
                    lhs.genotype, lhs.rstart,
                    lhs.rend) == std::tie(rhs.seq_id, rhs.pos, rhs.ref, rhs.alts, rhs.filter,
                                          rhs.info, rhs.qual, rhs.genotype, rhs.rstart, rhs.rend);
}

}  // namespace dorado::secondary
