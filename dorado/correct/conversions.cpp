#include "conversions.h"

#include <array>
#include <string>

namespace dorado::correction {

float normalize_quals(float q) { return 2.f * (q - MIN_QSCORE) / (MAX_QSCORE - MIN_QSCORE) - 1.f; }

std::array<char, 128> base_forward_mapping() {
    std::array<char, 128> base_forward = {0};
    base_forward['*'] = '*';
    base_forward['#'] = '*';
    base_forward['A'] = 'A';
    base_forward['a'] = 'A';
    base_forward['T'] = 'T';
    base_forward['t'] = 'T';
    base_forward['C'] = 'C';
    base_forward['c'] = 'C';
    base_forward['G'] = 'G';
    base_forward['g'] = 'G';
    return base_forward;
}

std::array<int, 128> gen_base_encoding() {
    std::array<int, 128> base_encoding = {0};
    const std::string bases = "ACGT*acgt#.";
    for (size_t i = 0; i < bases.length(); i++) {
        base_encoding[bases[i]] = static_cast<int>(i);
    }
    return base_encoding;
}

std::array<char, 11> gen_base_decoding() {
    std::array<char, 11> base_decoding = {0};
    const std::string bases = "ACGT*acgt#.";
    for (size_t i = 0; i < bases.length(); i++) {
        base_decoding[i] = bases[i];
    }
    return base_decoding;
}

}  // namespace dorado::correction
