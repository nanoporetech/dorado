#include "variant.h"

#include <ostream>

namespace dorado::polisher {

std::ostream& operator<<(std::ostream& os, const Variant& v) {
    os << v.seq_id << '\t' << v.pos << '\t' << v.ref << '\t' << v.alt << '\t' << v.filter << '\t'
       << v.qual;
    return os;
}

}  // namespace dorado::polisher
