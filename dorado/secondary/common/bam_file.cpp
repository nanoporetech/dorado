#include "secondary/common/bam_file.h"

#include <htslib/sam.h>
#include <spdlog/spdlog.h>

void HtsIdxDestructor::operator()(hts_idx_t* bam) { hts_idx_destroy(bam); }

namespace dorado::secondary {
BamFile::BamFile(const std::filesystem::path& in_fn)
        : m_fp{hts_open(in_fn.string().c_str(), "rb"), HtsFileDestructor()},
          m_idx{sam_index_load(m_fp.get(), in_fn.string().c_str()), HtsIdxDestructor()},
          m_hdr{sam_hdr_read(m_fp.get()), SamHdrDestructor()} {
    if (!m_fp) {
        throw std::runtime_error{"Could not open BAM file: '" + in_fn.string() + "'!"};
    }

    if (!m_idx) {
        throw std::runtime_error{"Could not open index for BAM file: '" + in_fn.string() + "'!"};
    }

    if (!m_hdr) {
        throw std::runtime_error{"Could not load header from BAM file: '" + in_fn.string() + "'!"};
    }
}

BamPtr BamFile::get_next() {
    BamPtr record(bam_init1(), BamDestructor());

    if (record == nullptr) {
        throw std::runtime_error{"Failed to initialize BAM record"};
        return BamPtr(nullptr, BamDestructor());
    }

    if (sam_read1(m_fp.get(), m_hdr.get(), record.get()) >= 0) {
        return record;
    }

    return BamPtr(nullptr, BamDestructor());
}

}  // namespace dorado::secondary