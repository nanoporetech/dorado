#include "types.h"

#include <htslib/sam.h>
#include <minimap.h>
#include <spdlog/spdlog.h>

namespace dorado {

std::shared_ptr<const BarcodingInfo> create_barcoding_info(
        const std::vector<std::string>& kit_names,
        bool barcode_both_ends,
        bool trim_barcode,
        BarcodingInfo::FilterSet allowed_barcodes,
        const std::optional<std::string>& custom_kit,
        const std::optional<std::string>& custom_seqs) {
    if (kit_names.empty() && !custom_kit) {
        return {};
    }

    // Use either the kit name, or the custom kit path as the "kit name" specifier since
    // the custom kit's name is not determined till the kit is parsed.
    std::string kit_name = "";
    if (!kit_names.empty()) {
        kit_name = kit_names[0];
    } else if (custom_kit.has_value()) {
        kit_name = *custom_kit;
    } else {
        throw std::runtime_error(
                "Neither kit name nor custom kit path was specified for BarcodeingInfo creation.");
    }
    spdlog::debug("Creating barcoding info for kit: {}", kit_name);
    auto result =
            BarcodingInfo{kit_name,   barcode_both_ends, trim_barcode, std::move(allowed_barcodes),
                          custom_kit, custom_seqs};
    return std::make_shared<const dorado::BarcodingInfo>(std::move(result));
}

void BamDestructor::operator()(bam1_t* bam) { bam_destroy1(bam); }

// Here mm_tbuf_t is used instead of mm_tbuf_s since minimap.h
// provides a typedef for mm_tbuf_s to mm_tbuf_t.
void MmTbufDestructor::operator()(mm_tbuf_t* tbuf) { mm_tbuf_destroy(tbuf); }

void SamHdrDestructor::operator()(sam_hdr_t* bam) { sam_hdr_destroy(bam); }

}  // namespace dorado
