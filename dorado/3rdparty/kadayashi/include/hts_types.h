#pragma once

#include <memory>

struct faidx_t;
struct htsFile;
struct hts_idx_t;
struct sam_hdr_t;
struct bam1_t;
struct hts_itr_t;
typedef struct htsFile samFile;
typedef struct sam_hdr_t bam_hdr_t;

namespace kadayashi {

struct FaidxDestructor {
    void operator()(faidx_t *) const noexcept;
};
using FaidxPtr = std::unique_ptr<faidx_t, FaidxDestructor>;

struct HtsFileDestructor {
    void operator()(htsFile *) const noexcept;
};
using HtsFilePtr = std::unique_ptr<htsFile, HtsFileDestructor>;

struct HtsIdxDestructor {
    void operator()(hts_idx_t *) const noexcept;
};
using HtsIdxPtr = std::unique_ptr<hts_idx_t, HtsIdxDestructor>;

struct SamHdrDestructor {
    void operator()(sam_hdr_t *) const noexcept;
};
using SamHdrPtr = std::unique_ptr<sam_hdr_t, SamHdrDestructor>;

struct BamDestructor {
    void operator()(bam1_t *) const noexcept;
};
using BamPtr = std::unique_ptr<bam1_t, BamDestructor>;

struct HtsItrDestructor {
    void operator()(hts_itr_t *) const noexcept;
};
using HtsItrPtr = std::unique_ptr<hts_itr_t, HtsItrDestructor>;

}  // namespace kadayashi