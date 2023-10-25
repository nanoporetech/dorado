#include "types.h"

#include "SampleSheet.h"

#include <htslib/sam.h>
#include <minimap.h>

namespace dorado {

std::shared_ptr<const BarcodingInfo> create_barcoding_info(
        const std::vector<std::string>& kit_names,
        bool barcode_both_ends,
        bool trim_barcode,
        std::unique_ptr<const utils::SampleSheet> sample_sheet) {
    if (kit_names.empty()) {
        return {};
    }

    auto result =
            BarcodingInfo{kit_names[0], barcode_both_ends, trim_barcode, std::move(sample_sheet)};
    return std::make_shared<const dorado::BarcodingInfo>(std::move(result));
}

void BamDestructor::operator()(bam1_t* bam) { bam_destroy1(bam); }

// Here mm_tbuf_t is used instead of mm_tbuf_s since minimap.h
// provides a typedef for mm_tbuf_s to mm_tbuf_t.
void MmTbufDestructor::operator()(mm_tbuf_t* tbuf) { mm_tbuf_destroy(tbuf); }

void SamHdrDestructor::operator()(sam_hdr_t* bam) { sam_hdr_destroy(bam); }

}  // namespace dorado
