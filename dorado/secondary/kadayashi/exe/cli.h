#pragma once

#include "secondary/kadayashi/types.h"

#include <filesystem>
#include <string>
#include <vector>

#define BAM_THREAD_RATIO 3
#define BAM_THREAD_MULT 4

#define KADAYASHI_VERSION "v0.1-r77"

struct cliopt_t {
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
    bool write_dbg_bam = false;

    // unphased pileup varcall hyperparameters
    kadayashi::pileup_pars_t pp;

    std::string interval_str;
    std::filesystem::path fn_tsv;  // not used by `phase`
};

struct cliopt_phaseglobal_t {
    int n_threads = 1;
    std::filesystem::path fn_ref;
    std::filesystem::path fn_bam;
    std::filesystem::path fn_vcf;
    std::filesystem::path output_name;
    std::filesystem::path fn_out_bam;
};

struct cliopt_varcall_t {
    std::filesystem::path output_prefix{"kadayashi"};
    std::filesystem::path fn_bam{};
    std::filesystem::path fn_ref{};

    int n_threads{1};
    bool write_dbg_bam{false};

    // unphased pileup varcall hyperparameters
    kadayashi::pileup_pars_t pp{};

    // varcall
    std::vector<std::string> varcall_regions{};
    int varcall_w{1'000'000};
    float max_gapcompressed_seqdiv{0.1f};
    bool ambig_ref{false};

    // output
    int bed_flanking{5000};
};

struct cliopt_featmatgen_t {
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

struct clio_pack_t {
    bool is_good{true};
    cliopt_t clio_phase;
    cliopt_phaseglobal_t clio_phaseglobal;
    cliopt_varcall_t clio_varcall;
    cliopt_featmatgen_t clio_featmatgen;
};

clio_pack_t parse_cli_main(int argc, char *argv[]);
