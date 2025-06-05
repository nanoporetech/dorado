#pragma once

#include "utils/types.h"

#include <filesystem>
#include <iosfwd>
#include <tuple>
#include <vector>

struct htsFile;
struct hts_idx_t;
struct sam_hdr_t;
struct bam1_t;

struct HtsIdxDestructor {
    void operator()(hts_idx_t*);
};
using HtsIdxPtr = std::unique_ptr<hts_idx_t, HtsIdxDestructor>;

namespace dorado::secondary {

struct HeaderLineData {
    std::string header_type;
    std::vector<std::pair<std::string, std::string>> tags;
};

class BamFile {
public:
    BamFile(const std::filesystem::path& in_fn);

    // Getters.
    htsFile* fp() const { return m_fp.get(); }
    hts_idx_t* idx() const { return m_idx.get(); }
    sam_hdr_t* hdr() const { return m_hdr.get(); }

    htsFile* fp() { return m_fp.get(); }
    hts_idx_t* idx() { return m_idx.get(); }
    sam_hdr_t* hdr() { return m_hdr.get(); }

    std::vector<HeaderLineData> parse_header() const;

    BamPtr get_next();

private:
    HtsFilePtr m_fp;
    HtsIdxPtr m_idx;
    SamHdrPtr m_hdr;
};

void header_to_stream(std::ostream& os, const std::vector<HeaderLineData>& header);

std::string header_to_string(const std::vector<HeaderLineData>& header);

}  // namespace dorado::secondary