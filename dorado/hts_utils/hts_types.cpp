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

}  // namespace

SamHdrSharedPtr::SamHdrSharedPtr(sam_hdr_t* raw) : m_header(make_sam_hdr(raw)) {};
SamHdrSharedPtr::SamHdrSharedPtr(SamHdrPtr hdr) : m_header(std::move(hdr)) {};

void HtsFileDestructor::operator()(htsFile* hts_file) {
    if (hts_file) {
        hts_close(hts_file);
    }
}

bool ReadAttributesComparator::operator()(const dorado::HtsData::ReadAttributes& lhs,
                                          const dorado::HtsData::ReadAttributes& rhs) const {
    // clang-format off
    return  lhs.sequencing_kit == rhs.sequencing_kit &&
            lhs.experiment_id == rhs.experiment_id &&
            lhs.sample_id == rhs.sample_id &&
            lhs.position_id == rhs.position_id &&
            lhs.flowcell_id == rhs.flowcell_id &&
            lhs.protocol_run_id == rhs.protocol_run_id &&
            lhs.acquisition_id == rhs.acquisition_id &&
            lhs.protocol_start_time_ms == rhs.protocol_start_time_ms &&
            lhs.subread_id == rhs.subread_id &&
            lhs.is_status_pass == rhs.is_status_pass;
    // clang-format on
};

size_t ReadAttributesHasher::operator()(const dorado::HtsData::ReadAttributes& attr) const {
    std::size_t seed = 0;
    hash_combine(seed, attr.sequencing_kit);
    hash_combine(seed, attr.experiment_id);
    hash_combine(seed, attr.sample_id);
    hash_combine(seed, attr.position_id);
    hash_combine(seed, attr.flowcell_id);
    hash_combine(seed, attr.protocol_run_id);
    hash_combine(seed, attr.acquisition_id);
    hash_combine(seed, attr.protocol_start_time_ms);
    hash_combine(seed, attr.subread_id);
    hash_combine(seed, attr.is_status_pass);
    return seed;
};

}  // namespace dorado
