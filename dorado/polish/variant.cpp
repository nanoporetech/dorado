#include "variant.h"

#include <ostream>

namespace dorado::polisher {

std::ostream& operator<<(std::ostream& os, const Variant& v) {
    os << v.seq_id << '\t' << v.pos << '\t' << v.ref << '\t' << v.alt << '\t' << v.filter << '\t'
       << v.qual << '\t' << v.rstart << '\t' << v.rend;
    return os;
}

bool operator==(const Variant& lhs, const Variant& rhs) {
    return std::tie(lhs.seq_id, lhs.pos, lhs.ref, lhs.alt, lhs.filter, lhs.info, lhs.qual,
                    lhs.genotype, lhs.rstart,
                    lhs.rend) == std::tie(rhs.seq_id, rhs.pos, rhs.ref, rhs.alt, rhs.filter,
                                          rhs.info, rhs.qual, rhs.genotype, rhs.rstart, rhs.rend);
}

}  // namespace dorado::polisher
