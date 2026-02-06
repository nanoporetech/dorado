#include "hts_utils/header_sq_record.h"

#include "utils/sequence_utils.h"

#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <cstdint>
#include <vector>

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

void MD5Generator::get_sequence_md5(MD5Hex& hex, const std::string& sequence) {
    hts_md5_reset(m_ctx);
    hts_md5_update(m_ctx, sequence.data(), static_cast<uint32_t>(sequence.size()));
    unsigned char digest[16];
    hts_md5_final(digest, m_ctx);
    hts_md5_hex(hex, digest);
}

}  // namespace dorado::utils