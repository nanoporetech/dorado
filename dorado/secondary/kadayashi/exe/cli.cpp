#include "cli.h"

#include "../string_utils.h"
#include "resources.h"

#include <ketopt.h>
#include <spdlog/spdlog.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cstdio>
#include <cstdlib>

ketopt_t opt;

// clang-format off
static ko_longopt_t longopts[] = {
        {"vcf", ko_required_argument,      301},  // vcf input for providing known variants; use --no-site-flt to disable filtering
        {"write-vcf", ko_no_argument,      302},  // write vcf of the initial variant calling
        {"bed", ko_required_argument,      303},  // bed input
        {"slice-in-bed", ko_no_argument,   304},  // whether to do sliding window in regions provided by bed
        // 305 vacant
        {"readtags", ko_required_argument, 306},  // input for `modifytag`
        {"readgroup", ko_required_argument, 307},
        {"region", ko_required_argument,   308},  // specify one region to work on, mostly a debug option. String considered 0-index.
        {"ref", ko_required_argument,      309},
        {"write-dbg-bam", ko_no_argument,  310},  // to be used with --region and automatically do `bin2bam`
        {"strict-intervals", ko_no_argument, 311},  // if set, do not allow expanding the requested interval(s)
        {"max-clipping", ko_required_argument, 312},  // reads with larger clippings will not contribute to informative site pileup
        {"chrom", ko_required_argument,    313},  // specify one chromosome
        // 314 vacant
        {"min-base-q", ko_required_argument,   401},
        {"min-site-cov", ko_required_argument, 402},
        // 403 vacant
        {"min-site-ratio", ko_required_argument,404},
        {"use-simple", ko_no_argument, 405},
        {"varcall-use-dvr", ko_no_argument, 406}, // note: bad CLI but don't break older scripts at least for now;
                                          // added for varcall to use dvr; mutually exclusive with 405
        {"verbose", ko_no_argument, 998},
        {"version", ko_no_argument, 999},
        {0, 0, 0}};
// clang-format on

int sancheck_cliopt(cliopt_t &clio) {
    if (clio.fn_bam.native().empty()) {
        spdlog::error("[kdys::{}] Missing input: must provide bam.", __func__);
        return 1;
    }
    if (!std::filesystem::exists(clio.fn_bam) || !std::filesystem::is_regular_file(clio.fn_bam)) {
        spdlog::error("[kdys::{}] Bad input: provided bam not found or is not a regular file.",
                      __func__);
        return 1;
    }

    if (clio.fn_ref.native().empty()) {
        spdlog::error("[kdys::{}] Missing input: must provide the reference genome.\n", __func__);
        return 1;
    }
    if (!std::filesystem::exists(clio.fn_ref) || !std::filesystem::is_regular_file(clio.fn_ref)) {
        spdlog::error(
                "[kdys::{}] Bad input: provided reference genome file not found or is not a "
                "regular "
                "file.",
                __func__);
        return 1;
    }

    if (clio.n_threads < 1) {
        spdlog::warn("[kdys::{}] clamping # threads to 1\n", __func__);
        clio.n_threads = 1;
    }
    if (clio.chunk_l < 10000) {
        spdlog::warn("[kdys::{}] clamping chunk size to 10kb\n", __func__);
        clio.chunk_l = 10000;
    }
    if (clio.chunk_stride < 100) {
        spdlog::warn("[kdys::{}] clamping chunk stride to 100bp\n", __func__);
        clio.chunk_stride = 100;
    }
    if (clio.pp.min_base_quality < 0) {
        spdlog::warn("[kdys::{}] min base qual was negative; clamping to 0\n", __func__);
        clio.pp.min_base_quality = 0;
    }
    if (clio.pp.min_varcall_coverage < 1) {
        spdlog::warn("[kdys::{}] min informative site coverage was set to <1; clamping to 1\n",
                     __func__);
        clio.pp.min_varcall_coverage = 1;
    }
    if (clio.pp.min_varcall_fraction < 0) {
        spdlog::warn(
                "[kdys::{}] min informative site coverage (ratio) was set to negative; clamping to "
                "0\n",
                __func__);
        clio.pp.min_varcall_fraction = 0;
    }
    if (clio.pp.min_varcall_fraction > 1) {
        spdlog::warn(
                "[kdys::{}] min informative site coverage (ratio) was set to >1; clamping to 1\n",
                __func__);
        clio.pp.min_varcall_fraction = 1;
    }
    if (clio.write_dbg_bam && clio.one_region_str.empty()) {
        spdlog::error("[kdys::{}] debug option --write-dbg-bam must be used with --region\n",
                      __func__);
        return 1;
    }
    if (clio.pp.max_clipping < 0) {
        spdlog::warn("[kdys::{}] max clipping size was negative; clamping to 0\n", __func__);
        clio.pp.max_clipping = 0;
    }
    return 0;
}

// clang-format off
void print_help_cliopt_t(cliopt_t &clio) {
    fprintf(stdout, "kadayashi %s\n", KADAYASHI_VERSION);
    fprintf(stdout,
            "Usage: kadayashi phase -o out --ref ref.fa "
            "[--bed fn_bed] [...] fn_bam 2>log\n");
    fprintf(stdout, "Outputs: $(out).tsv $(out).bin\n");
    fprintf(stdout, "Basic options:\n");
    fprintf(stdout, "  -o STR        [opt] Output file prefix. [%s]\n", clio.output_prefix.string().c_str());
    fprintf(stdout, "  --ref STR     [req] Rference genome. Must have fai index.\n");
    fprintf(stdout, "  -t INT        [opt] Number of threads. [%d]\n", clio.n_threads);
    fprintf(stdout, "  -c INT        [opt] Local haplotagging window size (chunk size) [%d]\n",
            clio.chunk_l);
    fprintf(stdout, "  -s INT        [opt] Local haplotagging stride (chunk stride) [%d]\n",
            clio.chunk_stride);
    fprintf(stdout, "  --verbose     [opt] Enable more output.\n");
    fprintf(stdout, "  --no-site-flt [opt] Disable strand and coverage-based filtering \n"
                    "                      for informative sites. Intended to be used \n"
                    "                      along --vcf to respect its variants.\n");
    fprintf(stdout, "  --bed STR      [opt] If provided, do local haplotagging in regions \n"
                    "                      specified by the bed file. Without --slice-in-bed, \n"
                    "                      will take entire intervals rather than doing \n"
                    "                      sliding windows (chunks) within them.\n");
    fprintf(stdout, "  --slice-in-bed [opt] Do sliding windows in bed-specified intervals.\n");
    fprintf(stdout, "  --region STR   [opt] Phase a single region without sliding window.\n");
    fprintf(stdout, "                       0-index [).\n");
    fprintf(stdout, "                       String can be like: chr6:1,100,50-2M\n");
    fprintf(stdout, "  --readgroup STR [opt] Specify a readgroup to include. Default is to include all.\n");
    fprintf(stdout, "  --write-dbg-bam[opt] Used with --region, automatically write haptagged bam w/ "
                    "index.\n");
    fprintf(stdout, "Hyperparameter options:\n");
    fprintf(stdout, "  --max-clipping INT [opt] Ignore reads with clippings larger than INT\n");
    fprintf(stdout, "                 on either side.[%d]\n", clio.pp.max_clipping);
    fprintf(stdout, "  --min-base-q INT   [opt] Ignore alt alleles with quality less than $INT. [%d] (TODO: "
                    "currently not affecting ref alleles.)\n", clio.pp.min_base_quality);
    fprintf(stdout, "  --min-site-cov INT [opt] Minimum informative site cov per allele. [%d]\n",
                    clio.pp.min_varcall_coverage);
    fprintf(stdout, "  --strict-intervals [opt] If set, do not allow attempts to expand\n");
    fprintf(stdout, "                           the requested interval(s).\n");
    fprintf(stdout, "  -h            [   ] Print this message and exit.\n");
}
// clang-format on

cliopt_t parse_cli(int argc, char *argv[]) {
    cliopt_t clio;
    opt = KETOPT_INIT;
    int c = 0;

    if (argc < 2) {
        print_help_cliopt_t(clio);
        clio.is_valid = false;
        return clio;
    }

    while ((c = ketopt(&opt, argc, argv, 1, "vho:t:s:c:i:", longopts)) >= 0) {
        if (c == 'h') {
            print_help_cliopt_t(clio);
            clio.is_print_help = true;
            return clio;
        } else if (c == 'o') {
            clio.output_prefix = std::filesystem::path(opt.arg);
        } else if (c == 't') {
            int nt = atoi(opt.arg);
            if (nt >= BAM_THREAD_RATIO) {
                clio.n_threads = nt / BAM_THREAD_RATIO;
            } else {
                clio.n_threads = nt;
            }
        } else if (c == 's') {
            clio.chunk_stride = atoi(opt.arg);
        } else if (c == 'c') {
            clio.chunk_l = atoi(opt.arg);
        } else if (c == 'i') {
            clio.interval_str = opt.arg;
        } else if (c == 301) {
            clio.fn_vcf = std::filesystem::path(opt.arg);
            spdlog::info(
                    "[kdys::{}] supplied vcf, will disable integrated varcall and use vcf variants "
                    "instead.\n",
                    __func__);
        } else if (c == 302) {
            clio.write_vcf = 1;
        } else if (c == 303) {
            clio.fn_bed = std::filesystem::path(opt.arg);
            spdlog::info("[kdys::{}] supplied bed, will do phasing only in these regions.\n",
                         __func__);
        } else if (c == 304) {
            clio.slice_in_bed = 1;
            spdlog::info("[kdys::{}] will use window & stride in bed regions.\n", __func__);
        } else if (c == 306) {
            clio.fn_tsv = std::filesystem::path(opt.arg);
        } else if (c == 307) {
            clio.pp.readgroup = opt.arg;
            spdlog::info("[kdys::{}] will use reads only with readgroup name {} .\n", __func__,
                         opt.arg);
        } else if (c == 308) {
            clio.one_region_str = opt.arg;
            spdlog::info("[kdys::{}] debug option - will only look at region {}\n", __func__,
                         opt.arg);
        } else if (c == 309) {
            clio.fn_ref = std::filesystem::path(opt.arg);
        } else if (c == 310) {
            clio.write_dbg_bam = 1;
        } else if (c == 311) {
            clio.pp.disable_region_expansion = true;
        } else if (c == 312) {
            clio.pp.max_clipping = atoi(opt.arg);
        } else if (c == 313) {
            clio.one_chrom_str = opt.arg;
            spdlog::info("[kdys::{}] debug option - will only look within reference: {}\n",
                         __func__, opt.arg);
        } else if (c == 401) {
            clio.pp.min_base_quality = atoi(opt.arg);
        } else if (c == 402) {
            clio.pp.min_varcall_coverage = atoi(opt.arg);
        } else if (c == 404) {
            clio.pp.min_varcall_fraction = static_cast<float>(atof(opt.arg));
        } else if (c == 405) {
            clio.use_simple = true;
            spdlog::info("[kdys::{}] using simple phasing\n", __func__);
        } else if (c == 998) {
            kadayashi::KDY_VERBOSE = true;
        } else if (c == 999) {
            fprintf(stdout, "kadayashi: %s\n", KADAYASHI_VERSION);
            clio.is_print_help = true;
            return clio;
        } else if (c == '?') {
            spdlog::error("[kdys::{}] unknown option argument in \"{}\"\n", __func__,
                          argv[opt.i - 1]);
            clio.is_valid = false;
            return clio;
        } else if (c == ':') {
            spdlog::error("[kdys::{}] missing option argument in \"{}\"\n", __func__,
                          argv[opt.i - 1]);
            clio.is_valid = false;
            return clio;
        }
    }
    if (argc - opt.ind > 1) {
        spdlog::error("[kdys::{}] invalid number of positional arguments ({})\n", __func__,
                      argc - opt.ind);
        print_help_cliopt_t(clio);
        clio.is_valid = false;
        return clio;
    }
    clio.fn_bam = std::filesystem::path(argv[opt.ind]);

    int bad_clio = sancheck_cliopt(clio);
    if (bad_clio) {
        clio.is_valid = false;
    }

    if (clio.fn_tsv.empty()) {
        clio.fn_tsv = clio.output_prefix;
        clio.fn_tsv += ".tsv";
    }
    return clio;
}

void print_help_tsv2bin_cli(cliopt_t &clio) {
    fprintf(stdout, "kadayashi %s\n", KADAYASHI_VERSION);
    fprintf(stdout, "Usage: kadayashi tsv2bin -o out.bin in.tsv\n");
    fprintf(stdout, "Outputs: out.bin\n");
    fprintf(stdout, "Options:\n");
    fprintf(stdout, "  -o STR        [req] Output file name. [%s]\n",
            clio.output_name.string().c_str());
}

cliopt_t parse_cli_tsv2bin(int argc, char *argv[]) {
    cliopt_t clio;
    opt = KETOPT_INIT;
    int c = 0;

    if (argc < 2) {
        print_help_tsv2bin_cli(clio);
        clio.is_print_help = true;
        return clio;
    }

    while ((c = ketopt(&opt, argc, argv, 1, "ho:t:r:", longopts)) >= 0) {
        if (c == 'h') {
            print_help_tsv2bin_cli(clio);
            clio.is_print_help = true;
            return clio;
        } else if (c == 'o') {
            clio.output_name = opt.arg;
        } else if (c == 't') {
            clio.n_threads = atoi(opt.arg);
        } else if (c == 'r') {
            clio.interval_str = opt.arg;
        } else if (c == 998) {
            kadayashi::KDY_VERBOSE = true;
        } else if (c == 999) {
            fprintf(stdout, "kadayashi: %s\n", KADAYASHI_VERSION);
            clio.is_print_help = true;
            return clio;
        } else if (c == '?') {
            spdlog::error("[kdys::{}] unknown option argument in \"{}\"\n", __func__,
                          argv[opt.i - 1]);
            clio.is_valid = false;
            return clio;
        } else if (c == ':') {
            spdlog::error("[kdys::{}] missing option argument in \"{}\"\n", __func__,
                          argv[opt.i - 1]);
            clio.is_valid = false;
            return clio;
        }
    }
    if (argc - opt.ind > 1) {
        spdlog::error("[kdys::{}] invalid number of positional arguments ({})\n", __func__,
                      argc - opt.ind);
        print_help_cliopt_t(clio);
        clio.is_valid = false;
        return clio;
    }
    clio.fn_tsv = std::filesystem::path(argv[opt.ind]);
    return clio;
}

static void print_help_bin2bam_cli(const cliopt_t &clio) {
    fprintf(stdout, "Usage: kadayashi bin2bam -o out.bam [-t threads] -r region fn_bam fn_bin\n");
    fprintf(stdout, "Options:\n");
    fprintf(stdout, "  fn_bam        Input bam.\n");
    fprintf(stdout, "  fn_bin        Kadayashi binary.\n");
    fprintf(stdout, "  -o STR        [req] Output bam name. Will create index. [%s]\n",
            clio.output_name.string().c_str());
    fprintf(stdout, "  -r/--region STR [req] Query region, 0-index [). [%s]\n",
            clio.one_region_str.c_str());
    fprintf(stdout, "  -t INT        [opt] Threads to use. [%s]\n",
            clio.output_name.string().c_str());
}
cliopt_t parse_cli_bin2bam(int argc, char *argv[]) {
    cliopt_t clio;
    opt = KETOPT_INIT;
    int c = 0;

    if (argc < 2) {
        print_help_bin2bam_cli(clio);
        clio.is_print_help = true;
        return clio;
    }

    while ((c = ketopt(&opt, argc, argv, 1, "vho:t:r:", longopts)) >= 0) {
        if (c == 'h') {
            print_help_bin2bam_cli(clio);
            return clio;
        } else if (c == 'o') {
            clio.output_name = opt.arg;
        } else if (c == 't') {
            clio.n_threads = atoi(opt.arg);
        } else if (c == 'r' || c == 308) {
            clio.one_region_str = opt.arg;
        } else if (c == 998) {
            kadayashi::KDY_VERBOSE = true;
        } else if (c == 999) {
            fprintf(stdout, "kadayashi: %s\n", KADAYASHI_VERSION);
            clio.is_print_help = true;
            return clio;
        } else if (c == '?') {
            spdlog::error("[kdys::{}] unknown option argument in \"{}\"\n", __func__,
                          argv[opt.i - 1]);
            clio.is_valid = false;
            return clio;
        } else if (c == ':') {
            spdlog::error("[kdys::{}] missing option argument in \"{}\"\n", __func__,
                          argv[opt.i - 1]);
            clio.is_valid = false;
            return clio;
        }
    }
    if (argc - opt.ind != 2) {
        spdlog::error("[kdys::{}] invalid number of positional arguments ({})\n", __func__,
                      argc - opt.ind);
        print_help_cliopt_t(clio);
        clio.is_valid = false;
        return clio;
    }
    clio.fn_bam = std::filesystem::path(argv[opt.ind]);
    clio.fn_bin = std::filesystem::path(argv[opt.ind + 1]);
    // simple sanchecks
    if (clio.one_region_str.empty()) {
        spdlog::error("[kdys::{}] need to provide interval string\n", __func__);
        clio.is_valid = false;
        return clio;
    }

    return clio;
}

static void print_help_phaseglobal_cli(const cliopt_phaseglobal_t &clio) {
    fprintf(stdout,
            "Usage: kadayashi phaseglobal -r fn_ref [-o output.vcf] [-t threads] fn_bam fn_vcf\n");
    fprintf(stdout, "Options:\n");
    fprintf(stdout, "  -r     STR [req] Reference genome, whose index must exist.\n");
    fprintf(stdout, "  fn_bam STR [req] Aligned, sorted & indexed reads.\n");
    fprintf(stdout, "  fn_vcf STR [req] Variants in VCF. Any existing phasing will be ignored.\n");
    fprintf(stdout,
            "  -o     STR [opt] Output VCF name. If not present, will be ${fn_vcf}.kadayashi.vcf "
            ".\n");
    fprintf(stdout, "  -t     INT [opt] Threads to use. [%d]\n", clio.n_threads);
    fprintf(stdout, "  -b     STR [opt] Write haptagged bam. Default to no output.\n");
}

cliopt_phaseglobal_t parse_cli_phaseglobal(int argc, char *argv[]) {
    cliopt_phaseglobal_t clio;
    opt = KETOPT_INIT;
    int c = 0;

    if (argc < 2) {
        print_help_phaseglobal_cli(clio);
        clio.is_print_help = true;
        return clio;
    }

    while ((c = ketopt(&opt, argc, argv, 1, "ho:t:r:b:", longopts)) >= 0) {
        if (c == 'h') {
            print_help_phaseglobal_cli(clio);
            clio.is_print_help = true;
            return clio;
        } else if (c == 'o') {
            clio.output_name = opt.arg;
        } else if (c == 'r') {
            clio.fn_ref = std::filesystem::path(opt.arg);
        } else if (c == 't') {
            clio.n_threads = atoi(opt.arg);
        } else if (c == 'b') {
            clio.fn_out_bam = std::filesystem::path(opt.arg);
        } else if (c == 998) {
            kadayashi::KDY_VERBOSE = true;
        } else if (c == 999) {
            fprintf(stdout, "kadayashi: %s\n", KADAYASHI_VERSION);
            clio.is_print_help = true;
            return clio;
        } else if (c == '?') {
            spdlog::error("[kdys::{}] unknown option argument in \"{}\"\n", __func__,
                          argv[opt.i - 1]);
            clio.is_valid = false;
            return clio;
        } else if (c == ':') {
            spdlog::error("[kdys::{}] missing option argument in \"{}\"\n", __func__,
                          argv[opt.i - 1]);
            clio.is_valid = false;
            return clio;
        }
    }
    if (argc - opt.ind != 2) {
        spdlog::error("[kdys::{}] invalid number of positional arguments ({})\n", __func__,
                      argc - opt.ind);
        print_help_phaseglobal_cli(clio);
        clio.is_valid = false;
        return clio;
    }
    clio.fn_bam = std::filesystem::path(argv[opt.ind]);
    clio.fn_vcf = std::filesystem::path(argv[opt.ind + 1]);

    // simple sanchecks
    if (clio.fn_ref.native().empty()) {
        spdlog::error("[kdys::{}] need to provide the reference genome\n", __func__);
        clio.is_valid = false;
        return clio;
    }
    if (clio.n_threads <= 0) {
        clio.n_threads = 1;
    }
    if (clio.output_name.native().empty()) {
        clio.output_name = clio.fn_vcf;
        clio.output_name += ".kadayashi.vcf";
    }

    return clio;
}

// clang-format off
static ko_longopt_t longopts_varcall[] = {
        {"region", ko_required_argument,   301},  // specify one region to work on, mostly a debug option. String considered 0-index.
        {"strict-intervals", ko_no_argument, 302},  // if set, do not allow expanding the requested interval(s)
        {"use-dvr", ko_no_argument, 303}, // note: bad CLI but don't break older scripts at least for now;
                                          // added for varcall to use dvr; mutually exclusive with 405
        {"ambig-ref", ko_no_argument, 304}, // varcall, include variants containing ambiguous reference bases
        {"max-gc-seqdiv", ko_required_argument, 305},  // varcall, max gap-compressed sequence divergence
        {"max-clipping", ko_required_argument, 306},  // reads with larger clippings will not contribute to informative site pileup
        {"min-strand-cov", ko_required_argument, 307},
        {"min-strand-cov-frac", ko_required_argument, 308},
        {"readgroup", ko_required_argument, 309},
        {"verbose", ko_no_argument, 998},
        {"version", ko_no_argument, 999},
        {0, 0, 0}};
// clang-format on

void print_help_varcall_cli(cliopt_varcall_t &clio) {
    fprintf(stdout, "kadayashi %s\n", KADAYASHI_VERSION);
    fprintf(stdout, "Usage: kadayashi varcall -o prefix [-t threads] [-b] fn_ref fn_bam\n");
    fprintf(stdout, "Options:\n");
    fprintf(stdout, "  fn_ref        Reference genome, index (.fai file).\n");
    fprintf(stdout, "  fn_bam        Input bam.\n");
    fprintf(stdout, "  -o STR        [req] Output prefix. [%s]\n",
            clio.output_prefix.string().c_str());
    fprintf(stdout, "  -t INT        [opt] Threads to use. [%d]\n", clio.n_threads);
    fprintf(stdout, "  -b            [opt] Set this to output phased bam. [not set]\n");
    fprintf(stdout, "  -w            [opt] Variant calling window size. [%d]\n", clio.varcall_w);
    fprintf(stdout,
            "  -p            [opt] Padding size for BED output. If specified multiple times, only "
            "the last value will be used. [%d]\n",
            clio.bed_flanking);
    fprintf(stdout,
            "  -r/--region   [opt] Specify region in 0-index [) with format like chr6:10M-11000000 "
            ".\n");
    fprintf(stdout, "                      Can use multiple times. Default is all.\n");
    fprintf(stdout, "  --use-dvr     [opt] Set to use dvr phasing method of simpel phasing. [%s]\n",
            clio.varcall_use_dvr ? "set" : "not set");
    fprintf(stdout,
            "  --min-strand-cov   [opt] Minimum coverage for both strands (inclusive). [%d]\n",
            clio.pp.min_strand_cov);
    fprintf(stdout,
            "  --min-strand-cov-frac   [opt] Minimum coverage fraction for both strands "
            "(inclusive). [%.3f]\n",
            clio.pp.min_strand_cov_frac);
    fprintf(stdout, "  --ambig-ref   [opt] Set to allow the output vcf to contain\n");
    fprintf(stdout, "                variants at non-ACGTU reference bases. [%s]\n",
            clio.ambig_ref ? "set" : "not set");
    fprintf(stdout, "  --strict-intervals [opt] Set to disable region expansion, i.e. \n");
    fprintf(stdout, "                      phase using strict the variants within \n");
    fprintf(stdout, "                      each chunk and don't look around. [%s]\n",
            clio.pp.disable_region_expansion ? "set" : "not set");
    fprintf(stdout, "  --max-gc-seqdiv [opt] Max gap-compressed sequence divergence allowed\n");
    fprintf(stdout, "                  for a read to be considered in phasing & varcall.[%.2f]\n",
            clio.pp.max_gapcompressed_seqdiv);
    fprintf(stdout, "  --max-clipping INT [opt] Ignore reads with clippings larger than INT\n");
    fprintf(stdout, "                     on either side.[%d]\n", clio.pp.max_clipping);
}

int sancheck_cliopt_varcall(cliopt_varcall_t &clio) {
    if (clio.fn_bam.native().empty()) {
        spdlog::error("[kdys::{}] Missing input: must provide bam.\n", __func__);
        return 1;
    }
    if (!std::filesystem::exists(clio.fn_bam) || !std::filesystem::is_regular_file(clio.fn_bam)) {
        spdlog::error("[kdys::{}] Bad input: provided bam not found or is not a regular file.",
                      __func__);
        return 1;
    }

    if (clio.fn_ref.native().empty()) {
        spdlog::error("[kdys::{}] Missing input: must provide the reference genome.\n", __func__);
        return 1;
    }
    if (!std::filesystem::exists(clio.fn_ref) || !std::filesystem::is_regular_file(clio.fn_ref)) {
        spdlog::error(
                "[kdys::{}] Bad input: provided reference genome file not found or is not a "
                "regular "
                "file.",
                __func__);
        return 1;
    }

    if (clio.n_threads < 1) {
        spdlog::warn("[kdys::{}] clamping # threads to 1\n", __func__);
        clio.n_threads = 1;
    }
    if (clio.pp.min_base_quality < 0) {
        spdlog::warn("[kdys::{}] min base qual was negative; clamping to 0\n", __func__);
        clio.pp.min_base_quality = 0;
    }
    if (clio.pp.min_varcall_coverage < 1) {
        spdlog::warn("[kdys::{}] min informative site coverage was set to <1; clamping to 1\n",
                     __func__);
        clio.pp.min_varcall_coverage = 1;
    }
    if (clio.pp.min_varcall_fraction < 0) {
        spdlog::warn(
                "[kdys::{}] min informative site coverage (ratio) was set to negative; clamping to "
                "0\n",
                __func__);
        clio.pp.min_varcall_fraction = 0;
    }
    if (clio.pp.min_varcall_fraction > 1) {
        spdlog::warn(
                "[kdys::{}] min informative site coverage (ratio) was set to >1; clamping to 1\n",
                __func__);
        clio.pp.min_varcall_fraction = 1;
    }
    if (clio.pp.max_clipping < 0) {
        spdlog::warn("[kdys::{}] max clipping size was negative; clamping to 0\n", __func__);
        clio.pp.max_clipping = 0;
    }
    if (clio.varcall_w < 10000) {
        spdlog::warn("[kdys::{}] window size too small, setting to 10kb instead\n", __func__);
        clio.varcall_w = 10000;
    }
    if (clio.bed_flanking <= 0) {
        spdlog::warn(
                "[kdys::{}] flanking size for output BED is too small, setting to 1 instead.\n",
                __func__);
        clio.bed_flanking = 1;
    }
    return 0;
}

cliopt_varcall_t parse_cli_varcall(int argc, char *argv[]) {
    cliopt_varcall_t clio;
    opt = KETOPT_INIT;
    int c = 0;

    if (argc < 2) {
        print_help_varcall_cli(clio);
        clio.is_print_help = true;
        return clio;
    }

    while ((c = ketopt(&opt, argc, argv, 1, "ht:o:br:w:p:", longopts_varcall)) >= 0) {
        if (c == 'h') {
            print_help_varcall_cli(clio);
            clio.is_print_help = true;
            return clio;
        } else if (c == 'o') {
            clio.output_prefix = std::filesystem::path(opt.arg);
        } else if (c == 't') {
            clio.n_threads = atoi(opt.arg);
        } else if (c == 'b') {
            clio.write_dbg_bam = 1;
        } else if (c == 'r' || c == 301) {  // add a region
            clio.varcall_regions.push_back(opt.arg);
        } else if (c == 'w') {  // window size (stride==windowsize)
            clio.varcall_w = atoi(opt.arg);
        } else if (c == 'p') {
            clio.bed_flanking = std::max(0, atoi(opt.arg));
        } else if (c == 302) {
            clio.pp.disable_region_expansion = true;
            spdlog::info("[kdys::{}] will not expand intervals during phasing\n", __func__);
        } else if (c == 303) {
            clio.varcall_use_dvr = true;
            spdlog::info("[kdys::{}] will use deepvariant replica phasing for read phasing\n",
                         __func__);
        } else if (c == 304) {
            clio.ambig_ref = true;
            fprintf(stdout,
                    "[M::%s] vcf output will include entries where reference allele contains "
                    "ambiguous bases\n",
                    __func__);
        } else if (c == 305) {
            clio.pp.max_gapcompressed_seqdiv = static_cast<float>(atof(opt.arg));
        } else if (c == 306) {
            clio.pp.max_clipping = atoi(opt.arg);
        } else if (c == 307) {
            clio.pp.min_strand_cov = atoi(opt.arg);
        } else if (c == 308) {
            clio.pp.min_strand_cov_frac = static_cast<float>(atof(opt.arg));
        } else if (c == 309) {
            clio.pp.readgroup = opt.arg;
            spdlog::info("[kdys::{}] will use reads only with readgroup name {} .\n", __func__,
                         opt.arg);
        } else if (c == 998) {
            kadayashi::KDY_VERBOSE = true;
        } else if (c == 999) {
            fprintf(stdout, "kadayashi: %s\n", KADAYASHI_VERSION);
            clio.is_print_help = true;
            return clio;
        } else if (c == '?') {
            spdlog::error("[kdys::{}] unknown option argument in \"{}\"\n", __func__,
                          argv[opt.i - 1]);
            clio.is_valid = false;
            return clio;
        } else if (c == ':') {
            spdlog::error("[kdys::{}] missing option argument in \"{}\"\n", __func__,
                          argv[opt.i - 1]);
            clio.is_valid = false;
            return clio;
        }
    }
    if (argc - opt.ind != 2) {
        spdlog::error("[kdys::{}] invalid number of positional arguments ({})\n", __func__,
                      argc - opt.ind);
        print_help_varcall_cli(clio);
        clio.is_valid = false;
        return clio;
    }

    clio.fn_ref = std::filesystem::path(argv[opt.ind]);
    clio.fn_bam = std::filesystem::path(argv[opt.ind + 1]);
    const int clio_parse_err = sancheck_cliopt_varcall(clio);
    if (clio_parse_err) {
        print_help_varcall_cli(clio);
        clio.is_valid = false;
        return clio;
    }

    return clio;
}

// clang-format off
static ko_longopt_t longopts_featmatgen[] = {
    {"exclude-dwells",        ko_no_argument,      301},
    {"exclude-haplotag",      ko_no_argument,      302},
    {"exclude-snp-qv",        ko_no_argument,      303},
    {"min-mapq",              ko_required_argument,304},
    {"readgroup",             ko_required_argument,305},
    {"disable-read-packing",  ko_no_argument,      306},
    {"max-lanes",             ko_required_argument,307},
    {"left-align-insertions", ko_no_argument,      308},
    {"min-snp-accuracy",      ko_required_argument,309},
    {"region",                ko_required_argument,310},
    {"use-medaka",            ko_no_argument,      311}
};
void print_help_cliopt_featmatgen_t(cliopt_featmatgen_t &clio) {
    fprintf(stdout, "kadayashi %s\n", KADAYASHI_VERSION);
    fprintf(stdout, "Usage: kadayashi featmatgen [-r ref:start-end -o fn_out -t threads] fn_ref fn_bam\n");
    fprintf(stdout, "Options:\n");
    fprintf(stdout, "  -r/--region   [opt] Generate feature matrix for the given region and write to file.\n");
    fprintf(stdout, "  -o            [opt] Output file name. Without -r no output will be written.\n");
    fprintf(stdout, "  --use-medaka  [opt] Use medaka's feature generator instead.\n");
    fprintf(stdout, "  -h            [   ] Print this message and exit.\n");
}
// clang-format on

cliopt_featmatgen_t parse_clio_featmatgen(int argc, char *argv[]) {
    cliopt_featmatgen_t clio;
    opt = KETOPT_INIT;
    int c = 0;
    if (argc < 2) {
        print_help_cliopt_featmatgen_t(clio);
        clio.is_valid = false;
        return clio;
    }

    while ((c = ketopt(&opt, argc, argv, 1, "ho:t:r:", longopts_featmatgen)) >= 0) {
        if (c == 'h') {
            print_help_cliopt_featmatgen_t(clio);
            clio.is_print_help = false;
            return clio;
        } else if (c == 'o') {
            clio.fn_out = opt.arg;
        } else if (c == 't') {
            bool ok = kadayashi::stoi_catch(clio.n_threads, opt.arg);
            if (!ok || clio.n_threads < 0) {
                spdlog::error("[kdys::{}] invalid argment from -t . Should be a positive integer.",
                              __func__);
                clio.is_valid = false;
                return clio;
            }
        } else if (c == 'r' || c == 110) {
            clio.itvl_str = opt.arg;
        } else if (c == 301) {
            clio.include_dwells = false;
        } else if (c == 302) {
            clio.include_haplotype_column = false;
        } else if (c == 303) {
            clio.include_snp_qv = false;
        } else if (c == 304) {
            bool ok = kadayashi::stoi_catch(clio.min_mapq, opt.arg);
            if (!ok || clio.min_mapq < 0) {
                spdlog::error(
                        "[kdys::{}] invalid argment from --min-mapq . Should be a positive "
                        "integer.",
                        __func__);
                clio.is_valid = false;
                return clio;
            }
        } else if (c == 305) {
            clio.readgroup = opt.arg;
        } else if (c == 306) {
            clio.disable_read_packing = true;
        } else if (c == 307) {
            bool ok = kadayashi::stoi_catch(clio.max_lanes, opt.arg);
            if (!ok || clio.max_lanes <= 0) {
                spdlog::error(
                        "[kdys::{}] invalid argment from --max-lanes. Should be a positive "
                        "integer.",
                        __func__);
                clio.is_valid = false;
                return clio;
            }
        } else if (c == 308) {
            clio.right_align_insertions = false;
        } else if (c == 309) {
            bool ok = kadayashi::stod_catch(clio.min_snp_accuracy, opt.arg);
            if (!ok || clio.min_snp_accuracy >= 1.0 || clio.min_snp_accuracy < 0) {
                spdlog::error(
                        "[kdys::{}] invalid argment from --min-snp-accuracy. Should be a double in "
                        "[0.0, 1.0).",
                        __func__);
                clio.is_valid = false;
                return clio;
            }
        } else if (c == 311) {
            clio.is_use_medaka = true;
        } else if (c == '?') {
            spdlog::error("[kdys::{}] unknown option argument in \"{}\"\n", __func__,
                          argv[opt.i - 1]);
            clio.is_valid = false;
            return clio;
        } else if (c == ':') {
            spdlog::error("[kdys::{}] missing option argument in \"{}\"\n", __func__,
                          argv[opt.i - 1]);
            clio.is_valid = false;
            return clio;
        }
    }
    if (argc - opt.ind != 2) {
        spdlog::error("[kdys::{}] invalid number of positional arguments ({})\n", __func__,
                      argc - opt.ind);
        print_help_cliopt_featmatgen_t(clio);
        clio.is_valid = false;
        return clio;
    }

    clio.fn_in_ref = argv[opt.ind];
    clio.fn_in_bam = argv[opt.ind + 1];

    return clio;
}
