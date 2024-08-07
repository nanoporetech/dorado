#include "alignment_utils.h"

#include <minimap.h>

#include <ostream>
#include <sstream>

namespace dorado::utils {

std::string alignment_to_str(const char* query,
                             const char* target,
                             const EdlibAlignResult& result) {
    std::stringstream ss;
    int tpos = result.startLocations[0];

    int qpos = 0;
    for (int i = 0; i < result.alignmentLength; i++) {
        if (result.alignment[i] == EDLIB_EDOP_DELETE) {
            ss << "-";
        } else {
            ss << query[qpos];
            qpos++;
        }
    }

    ss << '\n';

    for (int i = 0; i < result.alignmentLength; i++) {
        if (result.alignment[i] == EDLIB_EDOP_MATCH) {
            ss << "|";
        } else if (result.alignment[i] == EDLIB_EDOP_INSERT) {
            ss << " ";
        } else if (result.alignment[i] == EDLIB_EDOP_DELETE) {
            ss << " ";
        } else if (result.alignment[i] == EDLIB_EDOP_MISMATCH) {
            ss << "*";
        }
    }

    ss << '\n';

    for (int i = 0; i < result.alignmentLength; i++) {
        if (result.alignment[i] == EDLIB_EDOP_INSERT) {
            ss << "-";
        } else {
            ss << target[tpos];
            tpos++;
        }
    }

    return ss.str();
}

std::vector<CigarOp> parse_cigar_from_string(const std::string_view cigar) {
    std::vector<CigarOp> ops;
    ops.reserve(std::size(cigar));
    uint32_t len = 0;
    for (char c : cigar) {
        if (std::isdigit(c)) {
            len = len * 10 + (c - '0');
        } else {
            CigarOpType type;
            switch (c) {
            case '=':
                type = CigarOpType::EQ_MATCH;
                break;
            case 'X':
                type = CigarOpType::X_MISMATCH;
                break;
            case 'I':
                type = CigarOpType::INS;
                break;
            case 'D':
                type = CigarOpType::DEL;
                break;
            default:
                throw std::runtime_error("Unsupported CIGAR operation type " + std::string(1, c));
            }
            ops.emplace_back(CigarOp{type, len});
            len = 0;
        }
    }
    ops.shrink_to_fit();
    return ops;
}

std::vector<CigarOp> convert_mm2_cigar(const uint32_t* cigar, uint32_t n_cigar) {
    std::vector<dorado::CigarOp> cigar_ops;
    cigar_ops.resize(n_cigar);
    for (uint32_t i = 0; i < n_cigar; i++) {
        const uint32_t op = cigar[i] & 0xf;
        const uint32_t len = cigar[i] >> 4;

        // minimap2 --eqx must be set
        if (op == MM_CIGAR_EQ_MATCH) {
            cigar_ops[i] = {CigarOpType::EQ_MATCH, len};
        } else if (op == MM_CIGAR_X_MISMATCH) {
            cigar_ops[i] = {CigarOpType::X_MISMATCH, len};
        } else if (op == MM_CIGAR_INS) {
            cigar_ops[i] = {CigarOpType::INS, len};
        } else if (op == MM_CIGAR_DEL) {
            cigar_ops[i] = {CigarOpType::DEL, len};
        } else if (op == MM_CIGAR_MATCH) {
            throw std::runtime_error(
                    "cigar op MATCH is not supported must set minimap2 --eqx flag" +
                    std::to_string(op));
        } else {
            throw std::runtime_error("Unknown cigar op: " + std::to_string(op));
        }
    }
    return cigar_ops;
}

void serialize_cigar(std::ostream& os, const std::vector<CigarOp>& cigar) {
    for (auto& op : cigar) {
        char type = 'M';
        switch (op.op) {
        case CigarOpType::EQ_MATCH:
            type = '=';
            break;
        case CigarOpType::X_MISMATCH:
            type = 'X';
            break;
        case CigarOpType::INS:
            type = 'I';
            break;
        case CigarOpType::DEL:
            type = 'D';
            break;
        }
        os << op.len << type;
    }
}

std::string serialize_cigar(const std::vector<CigarOp>& cigar) {
    std::ostringstream oss;
    serialize_cigar(oss, cigar);
    return oss.str();
}

}  // namespace dorado::utils
