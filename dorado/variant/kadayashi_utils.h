#ifndef __KADAYASHI_UTILS_H__
#define __KADAYASHI_UTILS_H__

#include <stdint.h>
void write_binary_given_tsv(char *fn_tsv, char *fn_bin);
void *query_bin_file_get_qname2tag(char *fn_bin, char *chrom, uint32_t ref_start, uint32_t ref_end);

#endif  // __KADAYASHI_UTILS_H__