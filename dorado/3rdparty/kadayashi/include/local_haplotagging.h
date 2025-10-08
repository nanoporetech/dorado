#ifndef KADAYASHI_LOCALHAPTAG_H
#define KADAYASHI_LOCALHAPTAG_H

#include "types.h"

#include <memory>
#include <string>
#include <unordered_map>

struct faidx_t;
struct bam1_t;

namespace kadayashi {

int natoi(const char *s, const int l);

uint32_t max_of_u32_array(const uint32_t *a, const int l, int *idx);

phase_return_t kadayashi_local_haptagging_dvr_single_region1(samFile *fp_bam,
                                                             hts_idx_t *fp_bai,
                                                             sam_hdr_t *fp_header,
                                                             const faidx_t *fai,
                                                             const char *ref_name,
                                                             const uint32_t ref_start,
                                                             const uint32_t ref_end,
                                                             const pileup_pars_t &pp);

phase_return_t kadayashi_local_haptagging_simple_single_region1(samFile *fp_bam,
                                                                hts_idx_t *fp_bai,
                                                                sam_hdr_t *fp_header,
                                                                const faidx_t *fai,
                                                                const char *ref_name,
                                                                const uint32_t ref_start,
                                                                const uint32_t ref_end,
                                                                const pileup_pars_t &pp);

bool vg_gen(chunk_t *ck);

void vg_propogate(chunk_t *ck);

void vg_haptag_reads(chunk_t *ck);

void vg_do_simple_haptag(chunk_t *ck, uint32_t n_iter);

chunk_t unphased_varcall_a_chunk(bamfile_t *hf,
                                 ref_vars_t *refvars,
                                 const faidx_t *ref_faidx,
                                 const char *refname,
                                 const int32_t itvl_start,
                                 const int32_t itvl_end,
                                 const pileup_pars_t &pp,
                                 const int disable_lowcmp_mask);

std::unordered_map<std::string, int> kadayashi_dvr_single_region_wrapper(
        samFile *fp_bam,
        hts_idx_t *fp_bai,
        sam_hdr_t *fp_header,
        const faidx_t *fai,
        const std::string &ref_name,
        const uint32_t ref_start,
        const uint32_t ref_end,
        const int disable_interval_expansion,
        const int min_base_quality,
        const int min_varcall_coverage,
        const float min_varcall_fraction,
        const int varcall_indel_mask_flank,
        const int max_clipping,
        const int max_read_per_region);

}  // namespace kadayashi

#endif  //KADAYASHI_LOCALHAPTAG_H
