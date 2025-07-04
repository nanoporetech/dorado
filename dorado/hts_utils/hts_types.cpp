#include "hts_utils/hts_types.h"

#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <memory>

namespace dorado {

void BamDestructor::operator()(bam1_t* bam) { bam_destroy1(bam); }

void SamHdrDestructor::operator()(sam_hdr_t* bam) { sam_hdr_destroy(bam); }

SamHdrSharedPtr::SamHdrSharedPtr(sam_hdr_t* raw) : m_header(make(raw)) {};
SamHdrSharedPtr::SamHdrSharedPtr(SamHdrPtr hdr) : m_header(take(std::move(hdr))) {};

std::shared_ptr<const sam_hdr_t> SamHdrSharedPtr::make(sam_hdr_t* hdr) {
    return std::shared_ptr<const sam_hdr_t>(
            hdr, [](const sam_hdr_t* h) { sam_hdr_destroy(const_cast<sam_hdr_t*>(h)); });
}

std::shared_ptr<const sam_hdr_t> SamHdrSharedPtr::take(SamHdrPtr hdr) {
    return std::shared_ptr<const sam_hdr_t>(
            hdr.release(), [](const sam_hdr_t* h) { sam_hdr_destroy(const_cast<sam_hdr_t*>(h)); });
}

void HtsFileDestructor::operator()(htsFile* hts_file) {
    if (hts_file) {
        hts_close(hts_file);
    }
}

}  // namespace dorado
