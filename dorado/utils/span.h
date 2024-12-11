#pragma once

#include <cstddef>
#include <cstdint>

namespace dorado {

template <typename T>
class Span {
public:
    Span(T* data, const size_t size) noexcept : m_data{data}, m_size{size} {}

    const T& operator[](const size_t idx) const {
        assert(idx < m_size);
        return m_data[idx];
    }

    T* data() const { return m_data; }
    size_t size() const { return m_size; }
    int64_t ssize() const { return static_cast<int64_t>(m_size); }

private:
    T* m_data = nullptr;
    const size_t m_size = 0;
};

}  // namespace dorado
