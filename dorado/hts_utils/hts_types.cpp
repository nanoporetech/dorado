#include "hts_utils/hts_types.h"

#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <memory>

namespace dorado {

void BamDestructor::operator()(bam1_t* bam) { bam_destroy1(bam); }

void SamHdrDestructor::operator()(sam_hdr_t* bam) { sam_hdr_destroy(bam); }

namespace {

std::shared_ptr<const sam_hdr_t> make_sam_hdr(sam_hdr_t* hdr) {
    return std::shared_ptr<const sam_hdr_t>(
            hdr, [](const sam_hdr_t* h) { sam_hdr_destroy(const_cast<sam_hdr_t*>(h)); });
}

// Takes ownership of a sam_hdr_t from a unique pointer
std::shared_ptr<const sam_hdr_t> take_sam_hdr(SamHdrPtr hdr) {
    return std::shared_ptr<const sam_hdr_t>(
            hdr.release(), [](const sam_hdr_t* h) { sam_hdr_destroy(const_cast<sam_hdr_t*>(h)); });
}

}  // namespace

SamHdrSharedPtr::SamHdrSharedPtr(sam_hdr_t* raw) : m_header(make_sam_hdr(raw)) {};
SamHdrSharedPtr::SamHdrSharedPtr(SamHdrPtr hdr) : m_header(take_sam_hdr(std::move(hdr))) {};

void HtsFileDestructor::operator()(htsFile* hts_file) {
    if (hts_file) {
        hts_close(hts_file);
    }
}

}  // namespace dorado
