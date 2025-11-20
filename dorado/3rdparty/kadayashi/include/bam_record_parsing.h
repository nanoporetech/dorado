#pragma once

#include "blocked_bloom_filter.h"
#include "types.h"

#include <htslib/sam.h>

#include <cstdint>
#include <vector>

namespace kadayashi {

int sancheck_MD_tag_exists_and_is_valid(const bam1_t *aln);
void add_allele_qa_v(std::vector<qa_t> &h,
                     const uint32_t pos,
                     const char *allele,
                     const int allele_l,
                     const uint8_t cigar_op);
int parse_variants_for_one_read(const bam1_t *aln,
                                std::vector<qa_t> &vars,
                                const int min_base_qv,
                                int *left_clip_len,
                                int *right_clip_len,
                                const int SNPonly,
                                BlockedBloomFilter *bf);

}  // namespace kadayashi
