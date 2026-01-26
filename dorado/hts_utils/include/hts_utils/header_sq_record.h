
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct sam_hdr_t;
struct hts_md5_context;

namespace dorado::utils {

using MD5Hex = char[33];

// SAM SQ Header with optional UR and M5 tags
struct HeaderSQRecord {
    std::string sequence_name;                         // `@SQ SN`: Sequence Name
    uint32_t length;                                   // `@SQ LN`: Sequence length
    std::shared_ptr<const std::string> uri = nullptr;  // `@SQ UR`: URI of sequence.
    MD5Hex md5 = {};  // `@SQ M5`: MD5 sequence checksum (iff uri != nullptr)
};

class MD5Generator {
public:
    MD5Generator();
    ~MD5Generator();

    void get_sequence_md5(MD5Hex& hex, const std::string& sequence);
    void get_sequence_md5(MD5Hex& hex, const std::vector<uint8_t>& int_sequence);

private:
    hts_md5_context* m_ctx;
};

// Collection of SQ records
using HeaderSQRecords = std::vector<HeaderSQRecord>;

void add_sq_hdr(sam_hdr_t* hdr, const HeaderSQRecords& seqs);

}  // namespace dorado::utils
