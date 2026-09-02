#pragma once

#include "hts_utils/hts_types.h"
#include "secondary/common/alignment.h"

#include <filesystem>
#include <string>

struct htsFile;
struct hts_idx_t;
struct sam_hdr_t;
struct bam1_t;
struct hts_itr_t;

namespace dorado::secondary {

struct HtsIdxDestructor {
    void operator()(hts_idx_t*);
};
using HtsIdxPtr = std::unique_ptr<hts_idx_t, HtsIdxDestructor>;

struct HtsItrDestructor {
    void operator()(hts_itr_t* itr) const noexcept;
};
using HtsItrPtr = std::unique_ptr<hts_itr_t, HtsItrDestructor>;

struct BamFileView {
    htsFile* fp = nullptr;
    hts_idx_t* idx = nullptr;
    sam_hdr_t* hdr = nullptr;
};

class BamIterator {
public:
    BamIterator(hts_itr_t* itr, htsFile* fp);

    BamPtr get_next();

private:
    HtsItrPtr m_itr;
    htsFile* m_fp;
};

class BamFile {
public:
    BamFile(const std::filesystem::path& in_fn, int n_threads);

    // Getters.
    htsFile* fp() const { return m_fp.get(); }
    hts_idx_t* idx() const { return m_idx.get(); }
    sam_hdr_t* hdr() const { return m_hdr.get(); }

    htsFile* fp() { return m_fp.get(); }
    hts_idx_t* idx() { return m_idx.get(); }
    sam_hdr_t* hdr() { return m_hdr.get(); }

    BamPtr get_next();
    BamFileView get_view();

    BamIterator fetch(const std::string& chrom, int64_t start, int64_t end);

private:
    HtsFilePtr m_fp;
    HtsIdxPtr m_idx;
    SamHdrPtr m_hdr;
    int m_n_threads;
};

Alignment convert_bam1_to_aln(const bam1_t* b, const sam_hdr_t* hdr);

}  // namespace dorado::secondary
