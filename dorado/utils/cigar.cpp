#include "cigar.h"

#include <ostream>
#include <sstream>

namespace dorado {

std::ostream& operator<<(std::ostream& os, const CigarOp& a) {
    os << a.len << convert_cigar_op_to_char(a.op);
    return os;
}

std::string cigar_op_to_string(const CigarOp& a) {
    return std::to_string(a.len) + std::string(1, convert_cigar_op_to_char(a.op));
}

std::ostream& operator<<(std::ostream& os, const std::vector<CigarOp>& cigar) {
    for (CigarOp op : cigar) {
        os << op;
    }
    return os;
}

bool operator==(const CigarOp& a, const CigarOp& b) {
    return std::tie(a.op, a.len) == std::tie(b.op, b.len);
}

std::vector<CigarOp> parse_cigar_from_string(const std::string_view cigar) {
    std::vector<CigarOp> ops;
    ops.reserve(std::size(cigar));
    uint32_t len = 0;
    for (char c : cigar) {
        if (std::isdigit(c)) {
            len = len * 10 + (c - '0');
        } else {
            const CigarOpType op = CIGAR_CHAR_TO_OP[c];
            ops.emplace_back(CigarOp{op, len});
            len = 0;
        }
    }
    ops.shrink_to_fit();
    return ops;
}

std::vector<CigarOp> convert_mm2_cigar(const uint32_t* cigar, uint32_t n_cigar) {
    std::vector<CigarOp> cigar_ops;
    cigar_ops.resize(n_cigar);
    for (uint32_t i = 0; i < n_cigar; ++i) {
        const CigarOpType op = CIGAR_MM2_TO_DORADO[cigar[i] & 0xf];
        const uint32_t len = cigar[i] >> 4;
        cigar_ops[i] = {op, len};
    }
    return cigar_ops;
}

std::string serialize_cigar(const std::vector<CigarOp>& cigar) {
    std::ostringstream oss;
    oss << cigar;
    return oss.str();
}

}  // namespace dorado
