#include "hts_utils/FastxRandomReader.h"

#include "hts_utils/sequence_file_format.h"
#include "utils/string_utils.h"

#include <htslib/faidx.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <memory>

namespace dorado::hts_io {

void FaidxDestructor::operator()(faidx_t* faidx) { fai_destroy(faidx); }

namespace {
struct CharDestructor {
    void operator()(char* ptr) { hts_free(ptr); };
};

using CharPtr = std::unique_ptr<char, CharDestructor>;

FaidxPtr open_fai(const std::filesystem::path& fastx_path, const fai_format_options fmt) {
    FaidxPtr faidx_ptr(fai_load_format(fastx_path.string().c_str(), fmt));

    if (!faidx_ptr) {
        throw std::runtime_error{"Could not open .fai index file from path: '" +
                                 fastx_path.string() + "'."};
    }

    return faidx_ptr;
}
}  // namespace

FastxRandomReader::FastxRandomReader(const std::filesystem::path& fastx_path) {
    // Convert the string to lowercase.
    std::string path_str = fastx_path.string();
    std::transform(std::begin(path_str), std::end(path_str), std::begin(path_str),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    const hts_io::SequenceFormatType fmt = hts_io::parse_sequence_format(fastx_path);

    const fai_format_options fai_fmt = (fmt == hts_io::SequenceFormatType::FASTA)   ? FAI_FASTA
                                       : (fmt == hts_io::SequenceFormatType::FASTQ) ? FAI_FASTQ
                                                                                    : FAI_NONE;

    m_faidx = open_fai(fastx_path, fai_fmt);

    // Both attempts failed.
    if (!m_faidx) {
        spdlog::error("Could not create/load index for FASTx file {}", fastx_path.string());
        throw std::runtime_error("");
    }
}

std::string FastxRandomReader::fetch_seq(const std::string& read_id) const {
    int len = 0;
    CharPtr seq(fai_fetch(m_faidx.get(), read_id.c_str(), &len));
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
    CharPtr qual(fai_fetchqual(m_faidx.get(), read_id.c_str(), &len));
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
                       [](char c) { return static_cast<uint8_t>(c - 33); });
        return qscores;
    }
}

faidx_t* FastxRandomReader::get_raw_faidx_ptr() { return m_faidx.get(); }

int FastxRandomReader::num_entries() const { return faidx_nseq(m_faidx.get()); }

}  // namespace dorado::hts_io
