#include "faidx_utils.h"

#include <htslib/faidx.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace kadayashi::hts_utils {

namespace {
struct CharDestructor {
    void operator()(char* ptr) { hts_free(ptr); };
};

using CharPtr = std::unique_ptr<char, CharDestructor>;

}  // namespace

std::string fetch_seq(const faidx_t* fai, const std::string& region) {
    int len = 0;
    CharPtr seq(fai_fetch(fai, region.c_str(), &len));
    if (len == -2) {
        std::cerr << "Region not found: " << region;
        return {};
    } else if (len == -1) {
        throw std::runtime_error{"Could not fetch sequence for region " + region};
    } else {
        return std::string(seq.get(), seq.get() + len);
    }
}

int32_t fetch_seq_len(const faidx_t* fai, const std::string& seq_name) {
    const int32_t len = faidx_seq_len(fai, seq_name.c_str());
    if (len < 0) {
        std::cerr << "Could not fetch sequence length for " << seq_name;
    }
    return len;
}

std::vector<uint8_t> fetch_qual(const faidx_t* fai, const std::string& seq_name) {
    int len = 0;
    CharPtr qual(fai_fetchqual(fai, seq_name.c_str(), &len));
    if (len == -2) {
        std::cerr << "WARNING: Read qual " << seq_name << " not found\n";
        return {};
    } else if (len == -1) {
        throw std::runtime_error{"Could not fetch quality for " + seq_name};
    } else {
        std::vector<uint8_t> qscores;
        qscores.reserve(len);
        std::transform(qual.get(), qual.get() + len, std::back_inserter(qscores),
                       [](char c) { return static_cast<uint8_t>(c - 33); });
        return qscores;
    }
}

}  // namespace kadayashi::hts_utils
