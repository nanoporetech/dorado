#include "blocked_bloom_filter.h"

#include <cstring>
#include <memory>

namespace kadayashi {

uint64_t BlockedBloomFilter::m_bbf_hash64(uint32_t i) const { return std::hash<uint64_t>{}(i); }

BlockedBloomFilter::BlockedBloomFilter(int n_hashes, int n_blocks_bits)
        : m_is_enabled{false},
          m_is_frozen{false},
          m_n_hashes{n_hashes},
          m_n_blocks_bits{n_blocks_bits},
          m_mask_whichblock{(1ULL << n_blocks_bits) - 1},
          m_align_size_bytes{1 << (kadayashi::BBF_BLOCK_SHIFT - 3)},
          m_bbf_size_bytes{1ULL << (BBF_BLOCK_SHIFT + m_n_blocks_bits - 3)},
          m_data{nullptr} {}

void BlockedBloomFilter::enable() {
    if (!m_is_enabled) {
        m_data = static_cast<uint8_t *>(
                ::operator new(m_bbf_size_bytes, std::align_val_t{m_align_size_bytes}));
        std::memset(m_data, 0, m_bbf_size_bytes);
    } else {
        if (DEBUG_BBF_VERBOSE) {
            fprintf(stderr, "[E::%s] tried to enable bbf when it is already allocated\n", __func__);
        }
    }
    m_is_enabled = true;
}

BlockedBloomFilter::~BlockedBloomFilter() {
    if (m_is_enabled) {
        ::operator delete(m_data, std::align_val_t{m_align_size_bytes});
    }
}

bool BlockedBloomFilter::insert(uint32_t val) {
    // return: false if absent, true if exists ("is known")
    const uint64_t hash = m_bbf_hash64(val);
    const uint64_t whichblock = (hash & m_mask_whichblock) << (BBF_BLOCK_SHIFT - 3);
    uint8_t *block = &m_data[whichblock];
    int h1 = hash >> (m_n_blocks_bits + BBF_BLOCK_SHIFT);
    const int h2 = (h1 >> BBF_BLOCK_SHIFT) & BBF_BLOCK_MASK;
    h1 = h1 & BBF_BLOCK_MASK;
    int hi = h1;
    int cnt = 0;
    for (int i = 0; i < m_n_hashes; hi = (hi + h2) & BBF_BLOCK_MASK, i++) {
        uint8_t *byte = &block[hi >> 3];
        const uint8_t bit = 1 << (hi & 7);
        cnt += !!(bit & (*byte));
        *byte |= bit;
    }
    return cnt == m_n_hashes;
}

bool BlockedBloomFilter::query(uint32_t val) const {
    // Same as bbf_insert except that we do not modify the bf.
    // return: false if absent, true if exists ("is known")
    const uint64_t hash = m_bbf_hash64(val);
    const uint64_t whichblock = (hash & m_mask_whichblock) << (BBF_BLOCK_SHIFT - 3);
    const uint8_t *block = &m_data[whichblock];
    int h1 = hash >> (m_n_blocks_bits + BBF_BLOCK_SHIFT);
    const int h2 = (h1 >> BBF_BLOCK_SHIFT) & BBF_BLOCK_MASK;
    h1 = h1 & BBF_BLOCK_MASK;
    int hi = h1;
    int cnt = 0;
    for (int i = 0; i < m_n_hashes; hi = (hi + h2) & BBF_BLOCK_MASK, i++) {
        const uint8_t *byte = &block[hi >> 3];
        const uint8_t bit = 1 << (hi & 7);
        cnt += !!(bit & (*byte));
    }
    return cnt == m_n_hashes;
}

uintptr_t BlockedBloomFilter::get_data_address() const {
    return reinterpret_cast<uintptr_t>(m_data);
}
bool BlockedBloomFilter::is_frozen() const { return m_is_frozen; }

bool check_blockedbloomfilter_to_decide_inserting(BlockedBloomFilter *bf, uint32_t pos) {
    if (!bf) {
        return true;
    }
    const bool pos_is_known = bf->is_frozen() ? bf->query(pos) : bf->insert(pos);
    if (!pos_is_known) {
        return false;  // i.e. first time seeing the position, don't accept it yet.
    }
    return true;
}

}  // namespace kadayashi
