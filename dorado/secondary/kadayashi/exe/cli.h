#pragma once

#include "secondary/kadayashi/types.h"

#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#define BAM_THREAD_RATIO 3
#define BAM_THREAD_MULT 4

#define KADAYASHI_VERSION "v0.1-r76"

struct cliopt_t {
    bool is_valid = true;
    bool is_print_help = false;

    std::filesystem::path output_prefix = "kadayashi";
    std::filesystem::path output_name = "kadayashi.out";
    std::filesystem::path fn_bam;
    std::filesystem::path fn_vcf;
    std::filesystem::path fn_bed;
    std::filesystem::path fn_ref;
    std::filesystem::path fn_bin;

    std::string one_region_str;
    std::string one_chrom_str;
    int write_vcf = 0;
    int slice_in_bed = 0;
    int n_threads = 1;
    int chunk_l = 50'000;
    int chunk_stride = 30'000;
    int write_dbg_bam = 0;

    // unphased pileup varcall hyperparameters
    kadayashi::pileup_pars_t pp;

    // not-dvr phasing methods
    bool use_simple = false;  // toggle for: use simple phasing rather than dv reimpl

    std::string interval_str;
    std::filesystem::path fn_tsv;  // not used by `phase`
};
cliopt_t parse_cli(int argc, char *argv[]);
cliopt_t parse_cli_tsv2bin(int argc, char *argv[]);
cliopt_t parse_cli_bin2bam(int argc, char *argv[]);
int sancheck_cliopt(cliopt_t *opt);

struct cliopt_phaseglobal_t {
    bool is_valid = true;
    bool is_print_help = false;
    int n_threads = 1;
    std::filesystem::path fn_ref;
    std::filesystem::path fn_bam;
    std::filesystem::path fn_vcf;
    std::filesystem::path output_name;
    std::filesystem::path fn_out_bam;
};
cliopt_phaseglobal_t parse_cli_phaseglobal(int argc, char *argv[]);

struct cliopt_varcall_t {
    bool is_valid{true};
    bool is_print_help{false};

    std::filesystem::path output_prefix{"kadayashi"};
    std::filesystem::path fn_bam{};
    std::filesystem::path fn_ref{};

    int n_threads{1};
    int write_dbg_bam{0};

    // unphased pileup varcall hyperparameters
    kadayashi::pileup_pars_t pp{};

    // varcall
    bool varcall_use_dvr{false};
    std::vector<std::string> varcall_regions{};
    int varcall_w{1'000'000};
    float max_gapcompressed_seqdiv{0.1f};
    bool ambig_ref{false};

    // output
    int bed_flanking{5000};
};
void print_help_varcall_cli(cliopt_varcall_t &clio);
cliopt_varcall_t parse_cli_varcall(int argc, char *argv[]);

struct cliopt_featmatgen_t {
    bool is_valid{true};
    bool is_print_help{false};
    int n_threads{1};
    std::string itvl_str{};

    // IO
    std::filesystem::path fn_in_ref{};
    std::filesystem::path fn_in_bam{};
    std::filesystem::path fn_out{"kadayashi.featmatgen.out"};

    // toggle
    bool is_use_medaka{false};

    // to init MedakaFeatureMatrixOptions
    // (omitted: dypte, tag, hap_source)
    bool include_dwells{true};
    bool include_haplotype_column{true};
    bool include_snp_qv{true};
    int min_mapq{1};
    std::string readgroup{};
    bool disable_read_packing{false};  // see MedakaFeatureMatrixOptions comment
    int max_lanes{100};
    bool right_align_insertions{true};
    double min_snp_accuracy{0.0};
};

cliopt_featmatgen_t parse_clio_featmatgen(int argc, char *argv[]);
