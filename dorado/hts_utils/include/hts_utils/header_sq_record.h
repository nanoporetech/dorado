
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
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

    void get_sequence_md5(MD5Hex& hex, std::string_view sequence);

    static std::string fast_sequence_transform(std::string_view sequence);

private:
    MD5Generator(const MD5Generator&) = delete;
    MD5Generator& operator=(const MD5Generator&) = delete;
    MD5Generator(MD5Generator&&) = delete;
    MD5Generator& operator=(MD5Generator&&) = delete;

private:
    hts_md5_context* m_ctx;
};

// Collection of SQ records
using HeaderSQRecords = std::vector<HeaderSQRecord>;

void add_sq_hdr(sam_hdr_t* hdr, const HeaderSQRecords& seqs);

}  // namespace dorado::utils
