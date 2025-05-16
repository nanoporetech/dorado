#ifndef KADAYASHI_LOCALHAPTAG_H
#define KADAYASHI_LOCALHAPTAG_H

#include "types.h"

#include <string>
#include <unordered_map>

struct faidx_t;
struct bam1_t;

namespace kadayashi {

bamfile_t *init_bamfile_t_with_opened_files(samFile *fp_bam,
                                            hts_idx_t *fp_bai,
                                            sam_hdr_t *fp_header);

void destroy_holder_bamfile_t(bamfile_t *h, int include_self);

int natoi(const char *s, int l);

uint32_t max_of_u32_array(const uint32_t *a, int l, int *idx);

void *kadayashi_local_haptagging_dvr_single_region1(samFile *fp_bam,
                                                    hts_idx_t *fp_bai,
                                                    sam_hdr_t *fp_header,
                                                    faidx_t *fai,
                                                    const char *ref_name,
                                                    uint32_t ref_start,
                                                    uint32_t ref_end,
                                                    int disable_interval_expansion,
                                                    int min_base_quality,
                                                    int min_varcall_coverage,
                                                    float min_varcall_fraction,
                                                    int varcall_indel_mask_flank,
                                                    int max_clipping,
                                                    int max_read_per_region);

vg_t *vg_gen(chunk_t *ck);

void vg_propogate(vg_t *vg);

void vg_haptag_reads(vg_t *vg);

chunk_t *unphased_varcall_a_chunk(bamfile_t *hf,
                                  ref_vars_t *refvars,
                                  faidx_t *ref_faidx,
                                  const char *refname,
                                  int32_t itvl_start,
                                  int32_t itvl_end,
                                  int itvl_disable_expansion,
                                  int min_base_quality,
                                  int min_varcall_coverage,
                                  float min_varcall_coverage_ratio,
                                  int varcall_indel_mask_flank,
                                  int disable_info_site_filtering,
                                  int need_to_collect_compat,
                                  int max_clipping,
                                  int max_reads_in_chunk);

std::unordered_map<std::string, int32_t> kadayashi_dvr_single_region_wrapper(
        samFile *fp_bam,
        hts_idx_t *fp_bai,
        sam_hdr_t *fp_header,
        faidx_t *fai,
        const std::string &ref_name,
        uint32_t ref_start,
        uint32_t ref_end,
        int disable_interval_expansion,
        int min_base_quality,
        int min_varcall_coverage,
        float min_varcall_fraction,
        int varcall_indel_mask_flank,
        int max_clipping,
        int max_read_per_region);

}  // namespace kadayashi

#endif  //KADAYASHI_LOCALHAPTAG_H
