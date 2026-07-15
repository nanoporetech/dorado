#pragma once

#include "blocked_bloom_filter.h"
#include "types.h"

#include <cstdint>
#include <string_view>
#include <vector>

struct bam1_t;

namespace kadayashi {

extern const unsigned char md_op_table[256];

float get_tag_de_f(const bam1_t *aln);
bool to_exclude_by_flags(const bam1_t *aln, uint16_t unwanted_flags);
bool to_exclude_by_low_mapq(const bam1_t *aln, int min_mapq);
bool to_exlucde_by_high_de_tag(const bam1_t *aln, float max_gapcompressed_seqdiv);

bool sancheck_MD_tag_exists_and_is_valid(const bam1_t *aln);
void add_allele_qa_v(std::vector<qa_t> &h,
                     const uint32_t pos,
                     const std::string_view &allele,
                     const uint8_t cigar_op);
bool parse_variants_for_one_read(const bam1_t *aln,
                                 std::vector<qa_t> &vars,
                                 const int min_base_qv,
                                 int *left_clip_len,
                                 int *right_clip_len,
                                 const int SNPonly,
                                 BlockedBloomFilter *bf);

}  // namespace kadayashi
