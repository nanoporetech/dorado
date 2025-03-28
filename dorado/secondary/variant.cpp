#include "variant.h"

#include <ostream>
#include <tuple>

namespace dorado::secondary {

std::ostream& operator<<(std::ostream& os, const Variant& v) {
    os << v.seq_id << '\t' << v.pos << '\t' << v.ref << "\t{";
    for (size_t i = 0; i < std::size(v.alts); ++i) {
        if (i > 0) {
            os << ',';
        }
        os << '\'' << v.alts[i] << '\'';
    }
    os << "}\t" << v.filter << '\t' << v.qual << '\t' << v.rstart << '\t' << v.rend;

    {
        os << "\tgt:";
        bool first = true;
        for (const auto& [key, val] : v.genotype) {
            if (!first) {
                os << ',';
            }
            os << key << '=' << val;
            first = false;
        }
    }

    {
        os << "\tinfo:";
        bool first = true;
        for (const auto& [key, val] : v.info) {
            if (!first) {
                os << ',';
            }
            os << key << '=' << val;
            first = false;
        }
    }

    return os;
}

bool operator==(const Variant& lhs, const Variant& rhs) {
    return std::tie(lhs.seq_id, lhs.pos, lhs.ref, lhs.alts, lhs.filter, lhs.info, lhs.qual,
                    lhs.genotype, lhs.rstart,
                    lhs.rend) == std::tie(rhs.seq_id, rhs.pos, rhs.ref, rhs.alts, rhs.filter,
                                          rhs.info, rhs.qual, rhs.genotype, rhs.rstart, rhs.rend);
}

}  // namespace dorado::secondary
