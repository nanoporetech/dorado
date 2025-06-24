#include "hts_utils/hts_types.h"

#include <htslib/sam.h>
#include <minimap.h>
#include <spdlog/spdlog.h>

namespace dorado {

void BamDestructor::operator()(bam1_t* bam) { bam_destroy1(bam); }

void SamHdrDestructor::operator()(sam_hdr_t* bam) { sam_hdr_destroy(bam); }

void HtsFileDestructor::operator()(htsFile* hts_file) {
    if (hts_file) {
        hts_close(hts_file);
    }
}

// Here mm_tbuf_t is used instead of mm_tbuf_s since minimap.h
// provides a typedef for mm_tbuf_s to mm_tbuf_t.
void MmTbufDestructor::operator()(mm_tbuf_t* tbuf) { mm_tbuf_destroy(tbuf); }

}  // namespace dorado
