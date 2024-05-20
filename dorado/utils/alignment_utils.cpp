#include "alignment_utils.h"

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

std::vector<CigarOp> parse_cigar(std::string_view cigar) {
    std::vector<CigarOp> ops;
    std::string digits = "";
    for (char c : cigar) {
        if (std::isdigit(c)) {
            digits += c;
        } else {
            uint32_t len = std::atoi(digits.c_str());
            CigarOpType type;
            switch (c) {
            case 'M':
                type = CigarOpType::MATCH;
                break;
            case 'I':
                type = CigarOpType::INS;
                break;
            case 'D':
                type = CigarOpType::DEL;
                break;
            default:
                throw std::runtime_error("unknown type " + std::string(1, c));
            }
            digits = "";
            ops.push_back({type, len});
        }
    }
    return ops;
}

std::string serialize_cigar(const std::vector<CigarOp>& cigar) {
    std::stringstream ss;
    for (auto& op : cigar) {
        ss << op.len;
        char type = 'M';
        ;
        switch (op.op) {
        case CigarOpType::MATCH:
            type = 'M';
            break;
        case CigarOpType::MISMATCH:
            type = 'M';
            break;
        case CigarOpType::INS:
            type = 'I';
            break;
        case CigarOpType::DEL:
            type = 'D';
            break;
        }
        ss << type;
    }
    return ss.str();
}

}  // namespace dorado::utils
