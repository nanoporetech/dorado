#include "hts_utils/KString.h"

#include <htslib/sam.h>

#include <memory>

namespace dorado {

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
