#include "types.h"

#include <htslib/sam.h>
#include <minimap.h>
#include <spdlog/spdlog.h>

namespace dorado {

void BamDestructor::operator()(bam1_t* bam) { bam_destroy1(bam); }

// Here mm_tbuf_t is used instead of mm_tbuf_s since minimap.h
// provides a typedef for mm_tbuf_s to mm_tbuf_t.
void MmTbufDestructor::operator()(mm_tbuf_t* tbuf) { mm_tbuf_destroy(tbuf); }

void SamHdrDestructor::operator()(sam_hdr_t* bam) { sam_hdr_destroy(bam); }

void HtsFileDestructor::operator()(htsFile* hts_file) {
    if (hts_file) {
        hts_close(hts_file);
    }
}

KString::KString() : m_data(std::make_unique<kstring_t>()) { *m_data = {0, 0, nullptr}; }

KString::KString(size_t n) : m_data(std::make_unique<kstring_t>()) {
    *m_data = {0, 0, nullptr};
    ks_resize(m_data.get(), n);
}

KString::KString(kstring_t&& data) noexcept : m_data(std::make_unique<kstring_t>()) {
    *m_data = data;
    data = {0, 0, nullptr};
}

KString::KString(KString&& other) noexcept : m_data(std::make_unique<kstring_t>()) {
    *m_data = {0, 0, nullptr};
    m_data.swap(other.m_data);
}

KString& KString::operator=(KString&& rhs) noexcept {
    if (m_data->s) {
        ks_free(m_data.get());
    }
    m_data.swap(rhs.m_data);
    return *this;
}

KString::~KString() {
    if (m_data->s) {
        ks_free(m_data.get());
    }
}

kstring_t& KString::get() const { return *m_data; }

}  // namespace dorado
