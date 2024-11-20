#include "bam_file.h"

BamFile::BamFile(const std::filesystem::path &in_fn)
        : m_fp{hts_open(in_fn.c_str(), "rb"), hts_close},
          m_idx{nullptr, hts_idx_destroy},
          m_hdr{nullptr, sam_hdr_destroy} {
    if (!m_fp) {
        throw std::runtime_error{"Could not open BAM file: '" + in_fn.string() + "'!"};
    }

    m_idx = std::unique_ptr<hts_idx_t, decltype(&hts_idx_destroy)>(
            sam_index_load(m_fp.get(), in_fn.c_str()), hts_idx_destroy);

    if (!m_idx) {
        throw std::runtime_error{"Could not open index for BAM file: '" + in_fn.string() + "'!"};
    }

    m_hdr = std::unique_ptr<sam_hdr_t, decltype(&sam_hdr_destroy)>(sam_hdr_read(m_fp.get()),
                                                                   sam_hdr_destroy);

    if (!m_hdr) {
        throw std::runtime_error{"Could not load header from BAM file: '" + in_fn.string() + "'!"};
    }
}
