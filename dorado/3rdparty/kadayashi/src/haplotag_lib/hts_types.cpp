#include "hts_types.h"

#include <htslib/faidx.h>
#include <htslib/sam.h>

#include <stdexcept>

namespace kadayashi {

void FaidxDestructor::operator()(faidx_t* faidx) const noexcept { fai_destroy(faidx); }

void HtsIdxDestructor::operator()(hts_idx_t* bam) const noexcept { hts_idx_destroy(bam); }

void SamHdrDestructor::operator()(sam_hdr_t* bam) const noexcept { sam_hdr_destroy(bam); }

void HtsFileDestructor::operator()(htsFile* hts_file) const noexcept { hts_close(hts_file); }

void BamDestructor::operator()(bam1_t* bam) const noexcept { bam_destroy1(bam); }

void HtsItrDestructor::operator()(hts_itr_t* ptr) const noexcept { hts_itr_destroy(ptr); }

}  // namespace kadayashi