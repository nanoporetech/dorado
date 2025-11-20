#pragma once

#include <cstdint>

namespace kadayashi {

constexpr bool DEBUG_BBF_VERBOSE = false;
constexpr int32_t BBF_BLOCK_SHIFT = 9;
constexpr uint64_t BBF_BLOCK_MASK = (1ULL << BBF_BLOCK_SHIFT) - 1;

class BlockedBloomFilter {
public:
    BlockedBloomFilter(int n_hashes, int n_blocks_bits);
    ~BlockedBloomFilter();

    void enable();
    bool insert(uint32_t val);
    uintptr_t get_data_address() const;
    bool is_frozen() const;

    bool query(uint32_t val) const;

private:
    bool m_is_enabled;
    bool m_is_frozen;
    int m_n_hashes;
    int m_n_blocks_bits;
    uint64_t m_mask_whichblock;
    unsigned int m_align_size_bytes;
    uint64_t m_bbf_size_bytes;
    uint8_t *m_data;

    uint64_t m_bbf_hash64(uint32_t i) const;
};

bool check_blockedbloomfilter_to_decide_inserting(BlockedBloomFilter *bf, uint32_t pos);

}  // namespace kadayashi
