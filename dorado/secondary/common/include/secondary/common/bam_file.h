#pragma once

#include "hts_utils/hts_types.h"

#include <filesystem>
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

    BamPtr get_next();

private:
    HtsFilePtr m_fp;
    HtsIdxPtr m_idx;
    SamHdrPtr m_hdr;
};

}  // namespace dorado::secondary