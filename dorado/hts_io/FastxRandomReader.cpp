#include "FastxRandomReader.h"

#include <htslib/faidx.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <memory>

using CharPtr = std::unique_ptr<char[], void (*)(char*)>;

namespace dorado::hts_io {

void FaidxDestructor::operator()(faidx_t* faidx) { fai_destroy(faidx); }

FastxRandomReader::FastxRandomReader(const std::string& fastx_path) {
    auto faidx_ptr = fai_load_format(fastx_path.c_str(), FAI_FASTQ);
    if (!faidx_ptr) {
        spdlog::error("Could not create/load index for FASTx file {}", fastx_path);
        throw std::runtime_error("");
    }
    m_faidx.reset(faidx_ptr);
}

std::string FastxRandomReader::fetch_seq(const std::string& read_id) const {
    int len = 0;
    CharPtr seq(fai_fetch(m_faidx.get(), read_id.c_str(), &len), [](char* ptr) { delete[] ptr; });
    if (len == -2) {
        spdlog::error("Read {} not found", read_id);
        return "";
    } else if (len == -1) {
        spdlog::error("Could not fetch sequence for {}", read_id);
        throw std::runtime_error("");
    } else {
        return std::string(seq.get(), seq.get() + len);
    }
}

std::vector<uint8_t> FastxRandomReader::fetch_qual(const std::string& read_id) const {
    int len = 0;
    CharPtr qual(fai_fetchqual(m_faidx.get(), read_id.c_str(), &len),
                 [](char* ptr) { delete[] ptr; });
    if (len == -2) {
        spdlog::error("Read qual {} not found", read_id);
        return std::vector<uint8_t>();
    } else if (len == -1) {
        spdlog::error("Could not fetch quality for {}", read_id);
        throw std::runtime_error("");
    } else {
        std::vector<uint8_t> qscores;
        qscores.reserve(len);
        std::transform(qual.get(), qual.get() + len, std::back_inserter(qscores),
                       [](char c) { return static_cast<uint8_t>(c) - 33; });
        return qscores;
    }
}

int FastxRandomReader::num_entries() const { return faidx_nseq(m_faidx.get()); }

}  // namespace dorado::hts_io
