#pragma once

#include "secondary/common/bam_file.h"
#include "secondary/kadayashi/local_haplotagging.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace kadayashi {

int util_modify_tag_given_tsv(const std::filesystem::path &fn_bam,
                              const std::filesystem::path &fn_tsv,
                              const std::string &interval,
                              int n_threads,
                              int n_bam_threads,
                              const std::filesystem::path &fn_out);

struct region_string_t {
    // 1-based
    bool is_parse_success;
    bool is_whole_chrom;
    std::string chrom;
    uint32_t start;
    uint32_t end;
};
region_string_t parse_region_string2(std::string_view s);

int local_haplotagging(const std::filesystem::path &fn_bam,
                       const std::filesystem::path &fn_bed,
                       const int slice_in_bed,
                       const std::filesystem::path &fn_ref,
                       const std::string_view dbg_region_str,
                       const std::string_view dbg_chrom_str,
                       const std::filesystem::path &fn_vcf,
                       const int chunk_l,
                       const int chunk_stride,
                       const pileup_pars_t &pileup_pars,
                       const int n_threads,
                       const int n_bam_threads,
                       const int n_chunks_per_batch,
                       const std::filesystem::path &fn_out_tsv);

str2int_t kadayashi_global_phasing_simple_modify_vcf(const std::filesystem::path &fn_ref,
                                                     const std::filesystem::path &fn_bam,
                                                     const std::filesystem::path &fn_in_vcf,
                                                     const std::filesystem::path &fn_out_vcf,
                                                     const int n_threads);

str2int_t kadayashi_phased_variant_calling_threaded(const std::filesystem::path &fn_ref,
                                                    const std::filesystem::path &fn_bam,
                                                    const int n_threads,
                                                    const std::filesystem::path &fn_out_vcf,
                                                    const std::filesystem::path &prefix_out_unsr,
                                                    const int window_size,
                                                    const std::vector<std::string> &query_regions,
                                                    const std::string_view readgroup,
                                                    const int min_base_quality,
                                                    const int min_varcall_coverage,
                                                    const float min_varcall_fraction,
                                                    const int max_clipping,
                                                    const int min_strand_cov,
                                                    const float min_strand_cov_frac,
                                                    const float max_gapcompressed_seqdiv,
                                                    const bool ambig_ref,
                                                    const bool disable_interval_expansion,
                                                    const int bed_flanking);

intervals_t region_strings_to_intervals(dorado::secondary::BamFile &hf,
                                        const int window_size,
                                        const std::vector<std::string> &query_regions);

std::vector<std::string> bam_region_to_seqs(const std::filesystem::path &fn_ref,
                                            const std::filesystem::path &fn_bam,
                                            const std::string_view interval_string,
                                            const int n_threads,
                                            const std::filesystem::path &fn_out_fa);

}  // namespace kadayashi
