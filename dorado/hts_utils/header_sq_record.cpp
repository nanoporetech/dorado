#include "hts_utils/header_sq_record.h"

#include <htslib/sam.h>

#include <array>
#include <cstdint>
#include <numeric>

namespace {
constexpr std::array to_upper = []() {
    std::array<unsigned char, 256> lookup;
    std::iota(std::begin(lookup), std::end(lookup), 0);
    constexpr char offset = 'A' - 'a';
    for (char i = 'a'; i <= 'z'; ++i) {
        lookup[i] = i + offset;
    }
    return lookup;
}();
}  // namespace

namespace dorado::utils {

void add_sq_hdr(sam_hdr_t* hdr, const HeaderSQRecords& seqs) {
    for (const auto& s : seqs) {
        if (s.uri != nullptr) {
            sam_hdr_add_line(hdr, "SQ", "SN", s.sequence_name.c_str(), "LN",
                             std::to_string(s.length).c_str(), "M5", s.md5, "UR", s.uri->c_str(),
                             NULL);
        } else {
            sam_hdr_add_line(hdr, "SQ", "SN", s.sequence_name.c_str(), "LN",
                             std::to_string(s.length).c_str(), NULL);
        }
    }
}

MD5Generator::MD5Generator() : m_ctx(hts_md5_init()) {}
MD5Generator::~MD5Generator() { hts_md5_destroy(m_ctx); }

std::string MD5Generator::fast_sequence_transform(std::string_view sequence) {
    std::string transformed_sequence(sequence.size(), 0);
    size_t tr_idx = 0;
    for (size_t idx = 0; idx < sequence.size(); ++idx) {
        unsigned char c = static_cast<unsigned char>(sequence[idx]);
        if (c < 33 || c > 126) {
            continue;
        }
        transformed_sequence[tr_idx] = to_upper[c];
        ++tr_idx;
    }
    return std::move(transformed_sequence).substr(0, tr_idx);
}

void MD5Generator::get_sequence_md5(MD5Hex& hex, std::string_view sequence) {
    hts_md5_reset(m_ctx);
    std::string transformed_sequence = fast_sequence_transform(sequence);
    hts_md5_update(m_ctx, transformed_sequence.data(),
                   static_cast<uint32_t>(transformed_sequence.size()));
    unsigned char digest[16];
    hts_md5_final(digest, m_ctx);
    hts_md5_hex(hex, digest);
}

}  // namespace dorado::utils