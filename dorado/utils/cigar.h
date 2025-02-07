#pragma once

#include <array>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <tuple>
#include <vector>

namespace dorado {

// clang-format off
// Enum for handling CIGAR ops
enum class CigarOpType : uint8_t {
    M = 0,          // Alignment match. Can be either = or X.
    I = 1,          // Insertion.
    D = 2,          // Deletion.
    N = 3,          // Reference skip.
    S = 4,          // Soft clip.
    H = 5,          // Hard clip.
    P = 6,          // Padding.
    EQ = 7,         // Sequence match.
    X = 8,          // Sequence mismatch.
    UNDEFINED = 9,  // Not part of the CIGAR standard, needed for the lookup table.
};
// clang-format on

struct CigarOp {
    CigarOpType op{CigarOpType::UNDEFINED};
    uint32_t len{0};
};

constexpr std::array<CigarOpType, 256> CIGAR_CHAR_TO_OP = []() {
    std::array<CigarOpType, 256> lookup_table{};

    // Initialize all positions to CigarOpType::UNDEFINED.
    for (auto& elem : lookup_table) {
        elem = CigarOpType::UNDEFINED;
    }

    // Set the positions corresponding to each CIGAR operation.
    lookup_table['M'] = CigarOpType::M;
    lookup_table['I'] = CigarOpType::I;
    lookup_table['D'] = CigarOpType::D;
    lookup_table['N'] = CigarOpType::N;
    lookup_table['S'] = CigarOpType::S;
    lookup_table['H'] = CigarOpType::H;
    lookup_table['P'] = CigarOpType::P;
    lookup_table['='] = CigarOpType::EQ;
    lookup_table['X'] = CigarOpType::X;

    return lookup_table;
}();

constexpr std::array<CigarOpType, 256> CIGAR_MM2_TO_DORADO = []() {
    std::array<CigarOpType, 256> lookup_table{};

    // Initialize all positions to CigarOpType::UNDEFINED.
    for (auto& elem : lookup_table) {
        elem = CigarOpType::UNDEFINED;
    }

    // Set the positions corresponding to each CIGAR operation.
    lookup_table[0] = CigarOpType::M;
    lookup_table[1] = CigarOpType::I;
    lookup_table[2] = CigarOpType::D;
    lookup_table[3] = CigarOpType::N;
    lookup_table[4] = CigarOpType::S;
    lookup_table[5] = CigarOpType::H;
    lookup_table[6] = CigarOpType::P;
    lookup_table[7] = CigarOpType::EQ;
    lookup_table[8] = CigarOpType::X;

    return lookup_table;
}();

const std::array<char, 10> CIGAR_OP_TO_CHAR{
        'M', 'I', 'D', 'N', 'S', 'H', 'P', '=', 'X', 'U',
};

inline char convert_cigar_op_to_char(const CigarOpType op) {
    return CIGAR_OP_TO_CHAR[static_cast<int>(op)];
}

// Needs to be defined explicitly due to the enum class used for op.
bool operator==(const CigarOp& a, const CigarOp& b);

// Needed for the unit tests.
std::ostream& operator<<(std::ostream& os, const CigarOp& a);

std::string cigar_op_to_string(const CigarOp& a);

std::ostream& operator<<(std::ostream& os, const std::vector<CigarOp>& cigar);

std::vector<CigarOp> parse_cigar_from_string(const std::string_view cigar);

std::vector<CigarOp> convert_mm2_cigar(const uint32_t* cigar, uint32_t n_cigar);

void serialize_cigar(std::ostream& os, const std::vector<CigarOp>& cigar);

std::string serialize_cigar(const std::vector<CigarOp>& cigar);

}  // namespace dorado
