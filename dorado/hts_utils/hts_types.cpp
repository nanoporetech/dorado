#include "hts_utils/hts_types.h"

#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <memory>
#include <sstream>

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

bool HtsData::ReadAttributesCoreComparator::operator()(
        const dorado::HtsData::ReadAttributes& lhs,
        const dorado::HtsData::ReadAttributes& rhs) const {
    // Only using a subset ReadAttributes fields to index the core directory paths
    // clang-format off
    return  lhs.experiment_id == rhs.experiment_id &&
        lhs.sample_id == rhs.sample_id &&
        lhs.position_id == rhs.position_id &&
        lhs.flowcell_id == rhs.flowcell_id &&
        lhs.protocol_run_id == rhs.protocol_run_id &&
        lhs.protocol_start_time_ms == rhs.protocol_start_time_ms;
    // clang-format on
};

template <typename T>
void HtsData::ReadAttributesCoreHasher::hash_combine(std::size_t& seed, const T& value) const {
    seed ^= std::hash<T>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

size_t HtsData::ReadAttributesCoreHasher::operator()(
        const dorado::HtsData::ReadAttributes& attr) const {
    // Only using a subset ReadAttributes fields to index the core directory paths
    std::size_t seed = 0;
    hash_combine(seed, attr.experiment_id);
    hash_combine(seed, attr.sample_id);
    hash_combine(seed, attr.position_id);
    hash_combine(seed, attr.flowcell_id);
    hash_combine(seed, attr.protocol_run_id);
    hash_combine(seed, attr.protocol_start_time_ms);
    return seed;
};

std::string to_string(const HtsData::ReadAttributes& a) {
    std::ostringstream oss;
    oss << "{ ";
    oss << "kit:'" << a.sequencing_kit << "', ";
    oss << "exp:'" << a.experiment_id << "', ";
    oss << "sample:'" << a.sample_id << "', ";
    oss << "pos:'" << a.position_id << "', ";
    oss << "fc:'" << a.flowcell_id << "', ";
    oss << "proto:'" << a.protocol_run_id.substr(0, std::min(a.protocol_run_id.size(), size_t(8)))
        << "', ";
    oss << "acq:'" << a.acquisition_id.substr(0, std::min(a.acquisition_id.size(), size_t(8)))
        << "', ";
    oss << "st:'" << a.protocol_start_time_ms << "', ";
    oss << "subread:'" << a.subread_id << "', ";
    oss << "status:'" << a.is_status_pass << "'";
    oss << " }";
    return oss.str();
}

}  // namespace dorado
