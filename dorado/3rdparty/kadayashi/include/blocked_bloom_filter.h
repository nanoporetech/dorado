#pragma once

#include <cstdint>

namespace kadayashi {

constexpr int32_t BBF_BLOCK_SHIFT = 9;
constexpr uint64_t BBF_BLOCK_MASK = (1ULL << BBF_BLOCK_SHIFT) - 1;

class BlockedBloomFilter {
public:
    BlockedBloomFilter(int n_hashes, int n_blocks_bits);
    ~BlockedBloomFilter();

    bool enable();
    bool insert(uint32_t val);
    bool query(uint32_t val);

private:
    bool m_is_enabled;
    const int m_n_hashes;
    const int m_n_blocks_bits;
    const uint64_t m_mask_whichblock;
    const unsigned int m_align_size_bytes;
    const uint64_t m_bbf_size_bytes;
    uint8_t *m_data;

    uint64_t m_bbf_hash64(uint32_t i) const;
    bool find_or_insert_impl(uint32_t val, const bool is_insert);
};

bool check_blockedbloomfilter_to_decide_inserting(BlockedBloomFilter *bf, uint32_t pos);

}  // namespace kadayashi
