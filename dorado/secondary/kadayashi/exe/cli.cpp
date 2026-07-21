#include "cli.h"

#include "resources.h"

#include <argparse/argparse.hpp>
#include <spdlog/fmt/bundled/format.h>
#include <spdlog/spdlog.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <sstream>
#include <stdexcept>

void throws_sancheck_bam_is_valid(const std::filesystem::path &fn_bam) {
    if (fn_bam.native().empty()) {
        throw std::runtime_error(
                fmt::format("[kdys::{}] Missing input: must provide bam.", __func__));
    }
    if (!std::filesystem::exists(fn_bam) || !std::filesystem::is_regular_file(fn_bam)) {
        throw std::runtime_error(fmt::format(
                "[kdys::{}] Bad input: provided bam not found or is not a regular file.",
                __func__));
    }
}

void throws_sancheck_fastx_is_valid(const std::filesystem::path &fn_fastx) {
    if (fn_fastx.native().empty()) {
        throw std::runtime_error(fmt::format(
                "[kdys::{}] Missing input: must provide the reference genome.", __func__));
    }
    if (!std::filesystem::exists(fn_fastx) || !std::filesystem::is_regular_file(fn_fastx)) {
        throw std::runtime_error(fmt::format(
                "[kdys::{}] Bad input: provided reference genome file not found or is not a "
                "regular "
                "file.",
                __func__));
    }
}

void throws_sancheck_cliopt_phase(cliopt_t &clio) {
    throws_sancheck_bam_is_valid(clio.fn_bam);
    throws_sancheck_fastx_is_valid(clio.fn_ref);

    if (clio.n_threads < 1) {
        spdlog::warn("[kdys::{}] clamping # threads to 1", __func__);
        clio.n_threads = 1;
    }
    if (clio.chunk_l < 10000) {
        spdlog::warn("[kdys::{}] clamping chunk size to 10kb", __func__);
        clio.chunk_l = 10000;
    }
    if (clio.chunk_stride < 100) {
        spdlog::warn("[kdys::{}] clamping chunk stride to 100bp", __func__);
        clio.chunk_stride = 100;
    }
    if (clio.pp.min_base_quality < 0) {
        spdlog::warn("[kdys::{}] min base qual was negative; clamping to 0", __func__);
        clio.pp.min_base_quality = 0;
    }
    if (clio.pp.min_varcall_coverage < 1) {
        spdlog::warn("[kdys::{}] min informative site coverage was set to <1; clamping to 1",
                     __func__);
        clio.pp.min_varcall_coverage = 1;
    }
    if (clio.pp.min_varcall_fraction < 0) {
        spdlog::warn(
                "[kdys::{}] min informative site coverage (ratio) was set to negative; clamping to "
                "0",
                __func__);
        clio.pp.min_varcall_fraction = 0;
    }
    if (clio.pp.min_varcall_fraction > 1) {
        spdlog::warn(
                "[kdys::{}] min informative site coverage (ratio) was set to >1; clamping to 1",
                __func__);
        clio.pp.min_varcall_fraction = 1;
    }
    if (clio.write_dbg_bam && clio.one_region_str.empty()) {
        throw std::runtime_error(fmt::format(
                "[kdys::{}] debug option --write-dbg-bam must be used with --region", __func__));
    }
    if (clio.pp.max_clipping < 0) {
        spdlog::warn("[kdys::{}] max clipping size was negative; clamping to 0", __func__);
        clio.pp.max_clipping = 0;
    }
}

void parse_cli_phase_addarg(argparse::ArgumentParser &p, clio_pack_t &clio_pack) {
    const cliopt_t &clio = clio_pack.clio_phase;

    p.add_group("Required:");
    p.add_argument("--ref").required().help("Reference genome. Must have fai index.");
    p.add_argument("fn_bam").required().help("Input bam file. Must be sorted and index.");

    p.add_group("Frequently used:");
    p.add_argument("-o").default_value(clio.output_prefix.native()).help("Output file prefix.");
    p.add_argument("-t").scan<'i', int>().default_value(clio.n_threads).help("Number of threads.");
    p.add_argument("--bed").help(
            "If provided, do local haplotagging in regions specified by the bed file. "
            "Without --slice-in-bed, will take entire intervals rather than doing  sliding "
            "windows "
            "(chunks) within them. See also --slice-in-bed");
    p.add_argument("--slice-in-bed")
            .flag()
            .help("Do sliding windows in bed-specified intervals.")
            .action([func = __func__](const std::string &_) {
                spdlog::info("[kdys::{}] will use window & stride in bed regions.", func);
            });
    p.add_argument("--region")
            .help("Phase a single region without sliding window. 0-index [). String can be "
                  "like: chr6:1,100,50-2M");
    p.add_argument("--write-dbg-bam")
            .flag()
            .help("To be used with --region. When used, automatically write haptagged bam w/ "
                  "index.");
    p.add_argument("--strict-intervals")
            .flag()
            .help("If set, do not allow attempts to expand the requested interval(s).")
            .action([func = __func__](const std::string &_) {
                spdlog::info("[kdys::{}] region expansion disabled.", func);
            });
    p.add_argument("-i").help("Specify an interval string.");
    p.add_argument("--chrom").help("Debug option: will only look within the given reference.");

    p.add_group("Others:");
    p.add_argument("-c")
            .scan<'i', int>()
            .default_value(clio.chunk_l)
            .help("Local haplotagging window size (chunk size).");
    p.add_argument("-s")
            .scan<'i', int>()
            .default_value(clio.chunk_stride)
            .help("Local haplotagging stride (chunk stride).");
    p.add_argument("--max-clipping")
            .scan<'i', int>()
            .default_value(clio.pp.max_clipping)
            .help("Ignore reads with clippings larger than INT on either side.");
    p.add_argument("--min-base-q")
            .scan<'i', int>()
            .default_value(clio.pp.min_base_quality)
            .help("Ignore alt alleles with quality less than $INT. (TODO: currently does not "
                  "affect ref alleles.)");
    p.add_argument("--min-site-cov")
            .scan<'i', int>()
            .default_value(clio.pp.min_varcall_coverage)
            .help("Minimum informative site coverage per allele.");
    p.add_argument("--min-site-ratio")
            .scan<'f', float>()
            .default_value(clio.pp.min_varcall_fraction)
            .help("Minimum informative site coverage in fraction per allele. ");
    p.add_argument("--readgroup")
            .help("Specify a readgroup to include. Default is to include all.");
    p.add_argument("--vcf").help(
            "Supply a vcf with variants to be used for phasing. Only these variants will be "
            "used. (Default is to do a pileup-base variant calling.)");
    p.add_argument("--verbose").flag().help("Enable more output.");
}

void parse_cli_phase_fill(const argparse::ArgumentParser &p, clio_pack_t &clio_pack) {
    cliopt_t &clio = clio_pack.clio_phase;
    clio.fn_bam = p.get("fn_bam");
    clio.fn_ref = std::filesystem::path(p.get("--ref"));
    clio.chunk_stride = p.get<int>("-s");
    clio.chunk_l = p.get<int>("-c");
    clio.pp.max_clipping = p.get<int>("--max-clipping");
    clio.pp.min_base_quality = p.get<int>("--min-base-q");
    clio.pp.min_varcall_coverage = p.get<int>("--min-site-cov");
    clio.pp.min_varcall_fraction = p.get<float>("--min-site-ratio");

    clio.output_prefix = std::filesystem::path(p.get<std::string>("-o"));
    {
        int nt = p.get<int>("-t");
        if (nt >= BAM_THREAD_RATIO) {
            clio.n_threads = nt / BAM_THREAD_RATIO;
        } else {
            clio.n_threads = nt;
        }
    }
    if (const auto v = p.present<std::string>("-i"); v) {
        clio.interval_str = v.value();
    }
    if (const auto v = p.present<std::string>("--vcf"); v) {
        clio.fn_vcf = std::filesystem::path(v.value());
        spdlog::info(
                "[kdys::{}] supplied vcf, will disable integrated varcall and use vcf variants "
                "instead.",
                __func__);
    }
    if (const auto v = p.present<std::string>("--bed"); v) {
        clio.fn_bed = std::filesystem::path(v.value());
        spdlog::info("[kdys::{}] supplied bed, will do phasing only in these regions.", __func__);
    }
    if (const auto v = p.present<std::string>("--readgroup"); v) {
        clio.pp.readgroup = v.value();
        spdlog::info("[kdys::{}] will use reads only with readgroup name {} .", __func__,
                     clio.pp.readgroup);
    }
    if (const auto v = p.present<std::string>("--region"); v) {
        clio.one_region_str = v.value();
    }
    if (const auto v = p.present<std::string>("--chrom"); v) {
        clio.one_chrom_str = v.value();
    }
    clio.slice_in_bed = p.get<bool>("--slice-in-bed");
    clio.write_dbg_bam = p.get<bool>("--write-dbg-bam");
    clio.pp.disable_region_expansion = p.get<bool>("--strict-intervals");
    kadayashi::KDY_VERBOSE = p.get<bool>("--verbose");

    throws_sancheck_cliopt_phase(clio);
    if (clio.fn_tsv.empty()) {
        clio.fn_tsv = clio.output_prefix;
        clio.fn_tsv += ".tsv";
    }
}

void parse_cli_modifytag_addarg(argparse::ArgumentParser &p, clio_pack_t &clio_pack) {
    const cliopt_t &clio = clio_pack.clio_phase;
    p.add_argument("fn_bam").required().help("Input bam.");
    p.add_argument("--readtags")
            .required()
            .help("2-col tsv mapping qname to haptag (0-index), not kadayashi tsv");
    p.add_argument("-o").default_value(clio.output_prefix.native()).help("Output prefix.");
    p.add_argument("-i").help("Specify an interval string.");
    p.add_argument("-t").scan<'i', int>().default_value(clio.n_threads).help("Number of threads.");
    p.add_argument("--verbose").flag().help("Enable more output.");
}

void parse_cli_modifytag_fill(const argparse::ArgumentParser &p, clio_pack_t &clio_pack) {
    cliopt_t &clio = clio_pack.clio_phase;
    clio.fn_bam = p.get("fn_bam");
    clio.fn_tsv = p.get("--readtags");

    clio.output_prefix = p.get<std::string>("-o");
    clio.n_threads = p.get<int>("-t");

    if (const auto v = p.present<std::string>("-i"); v) {
        clio.interval_str = v.value();
    }
    kadayashi::KDY_VERBOSE = p.get<bool>("--verbose");
}

void parse_cli_tsv2bin_addarg(argparse::ArgumentParser &p, clio_pack_t &clio_pack) {
    const cliopt_t &clio = clio_pack.clio_phase;
    p.add_argument("fn_tsv").required();
    p.add_argument("-o").default_value(clio.output_name.native()).help("Output file name.");
    p.add_argument("-t").scan<'i', int>().default_value(clio.n_threads).help("Number of threads.");
    p.add_argument("-r", "--region").help("Specify an internal string");
    p.add_argument("--verbose").flag().help("Enable more output.");
}

void parse_cli_tsv2bin_fill(const argparse::ArgumentParser &p, clio_pack_t &clio_pack) {
    cliopt_t &clio = clio_pack.clio_phase;
    clio.fn_tsv = p.get("fn_tsv");

    clio.output_name = p.get<std::string>("-o");
    {
        clio.n_threads = p.get<int>("-t");
        if (clio.n_threads < 1) {
            spdlog::warn("[kdys::{}] clamping # threads to 1", __func__);
            clio.n_threads = 1;
        }
    }
    if (const auto v = p.present<std::string>("--region"); v) {  // or -r
        clio.interval_str = v.value();
    }
    kadayashi::KDY_VERBOSE = p.get<bool>("--verbose");
}

void parse_cli_bin2bam_addarg(argparse::ArgumentParser &p, clio_pack_t &clio_pack) {
    const cliopt_t &clio = clio_pack.clio_phase;

    p.add_argument("fn_bam").required().help("Input bam");
    p.add_argument("fn_bin").required().help("Kadayashi bin file.");
    p.add_argument("-r", "--region").required().help("Query region, 0-index [).");
    p.add_argument("-o")
            .default_value(clio.output_name.native())
            .help("Output bam name. Will be sorted and create index.");
    p.add_argument("-t").scan<'i', int>().default_value(clio.n_threads).help("Number of threads.");
    p.add_argument("--verbose").flag().help("Enable more output.");
}

void parse_cli_bin2bam_fill(const argparse::ArgumentParser &p, clio_pack_t &clio_pack) {
    cliopt_t &clio = clio_pack.clio_phase;
    clio.fn_bam = p.get("fn_bam");
    clio.fn_bin = p.get("fn_bin");

    clio.output_name = p.get<std::string>("-o");

    if (const auto v = p.present<std::string>("--region"); v) {  // or -r
        clio.one_region_str = v.value();
    }

    clio.n_threads = p.get<int>("-t");
    if (clio.n_threads < 1) {
        spdlog::warn("[kdys::{}] clamping # threads to 1", __func__);
        clio.n_threads = 1;
    }
    kadayashi::KDY_VERBOSE = p.get<bool>("--verbose");

    // some sancheck
    throws_sancheck_bam_is_valid(clio.fn_bam);
    if (clio.fn_bin.native().empty()) {
        throw std::runtime_error(fmt::format("[kdys::{}] must provide a bin file name.", __func__));
    }
    if (clio.one_region_str.empty()) {
        throw std::runtime_error(
                fmt::format("[kdys::{}] need to provide interval string.", __func__));
    }
}

void parse_cli_phaseglobal_addarg(argparse::ArgumentParser &p, clio_pack_t &clio_pack) {
    // global phasing
    const cliopt_phaseglobal_t &clio = clio_pack.clio_phaseglobal;

    p.add_argument("fn_bam").required().help("Input bam.");
    p.add_argument("fn_vcf").required().help(
            "Variants in VCF. Any existing phasing will be ignored.");
    p.add_argument("-r").required().help("Reference genome. Index must exist.");
    p.add_argument("-o").required().help("Output VCF file name.");
    p.add_argument("-t").scan<'i', int>().default_value(clio.n_threads).help("Threads to use.");
    p.add_argument("-b").help("Filename. If given, write a haptagged bam file under the filename.");
    p.add_argument("--verbose").flag().help("Enable more output.");
}

void parse_cli_phaseglobal_fill(const argparse::ArgumentParser &p, clio_pack_t &clio_pack) {
    cliopt_phaseglobal_t &clio = clio_pack.clio_phaseglobal;
    clio.fn_bam = p.get("fn_bam");
    clio.fn_vcf = p.get("fn_vcf");
    clio.fn_ref = p.get("-r");
    clio.output_name = p.get<std::string>("-o");

    clio.n_threads = p.get<int>("-t");

    if (const auto v = p.present<std::string>("-b"); v) {
        clio.fn_out_bam = std::filesystem::path(v.value());
    }
    kadayashi::KDY_VERBOSE = p.get<bool>("--verbose");

    throws_sancheck_bam_is_valid(clio.fn_bam);
    throws_sancheck_fastx_is_valid(clio.fn_ref);
    if (clio.fn_vcf.native().empty()) {
        throw std::runtime_error(fmt::format("[kdys::{}] must provide a VCF file.", __func__));
    }

    // simple sanchecks
    if (clio.n_threads <= 0) {
        clio.n_threads = 1;
    }
    if (clio.output_name.native().empty()) {
        clio.output_name = clio.fn_vcf;
        clio.output_name += ".kadayashi.vcf";
    }
}

void throws_sancheck_cliopt_varcall(cliopt_varcall_t &clio) {
    if (clio.fn_bam.native().empty()) {
        throw std::runtime_error(
                fmt::format("[kdys::{}] Missing input: must provide bam.", __func__));
    }
    if (!std::filesystem::exists(clio.fn_bam) || !std::filesystem::is_regular_file(clio.fn_bam)) {
        throw std::runtime_error(fmt::format(
                "[kdys::{}] Bad input: provided bam not found or is not a regular file.",
                __func__));
    }

    if (clio.fn_ref.native().empty()) {
        throw std::runtime_error(fmt::format(
                "[kdys::{}] Missing input: must provide the reference genome.", __func__));
    }
    if (!std::filesystem::exists(clio.fn_ref) || !std::filesystem::is_regular_file(clio.fn_ref)) {
        throw std::runtime_error(fmt::format(
                "[kdys::{}] Bad input: provided reference genome file not found or is not a "
                "regular "
                "file.",
                __func__));
    }

    if (clio.n_threads < 1) {
        spdlog::warn("[kdys::{}] clamping # threads to 1", __func__);
        clio.n_threads = 1;
    }
    if (clio.pp.min_base_quality < 0) {
        spdlog::warn("[kdys::{}] min base qual was negative; clamping to 0", __func__);
        clio.pp.min_base_quality = 0;
    }
    if (clio.pp.min_varcall_coverage < 1) {
        spdlog::warn("[kdys::{}] min informative site coverage was set to <1; clamping to 1",
                     __func__);
        clio.pp.min_varcall_coverage = 1;
    }
    if (clio.pp.min_varcall_fraction < 0) {
        throw std::runtime_error(fmt::format(
                "[kdys::{}] min informative site coverage (ratio) was set to negative; clamping to "
                "0",
                __func__));
        clio.pp.min_varcall_fraction = 0;
    }
    if (clio.pp.min_varcall_fraction > 1) {
        spdlog::warn(
                "[kdys::{}] min informative site coverage (ratio) was set to >1; clamping to 1",
                __func__);
        clio.pp.min_varcall_fraction = 1;
    }
    if (clio.pp.max_clipping < 0) {
        spdlog::warn("[kdys::{}] max clipping size was negative; clamping to 0", __func__);
        clio.pp.max_clipping = 0;
    }
    if (clio.varcall_w < 10000) {
        spdlog::warn("[kdys::{}] window size too small, setting to 10kb instead", __func__);
        clio.varcall_w = 10000;
    }
    if (clio.bed_flanking <= 0) {
        spdlog::warn("[kdys::{}] flanking size for output BED is too small, setting to 1 instead.",
                     __func__);
        clio.bed_flanking = 1;
    }
}

void parse_cli_varcall_addarg(argparse::ArgumentParser &p, clio_pack_t &clio_pack) {
    // pileup-based variant calling
    const cliopt_varcall_t &clio = clio_pack.clio_varcall;

    p.add_group("Required:");
    p.add_argument("fn_ref").required().help("Reference genome. Index must exist.");
    p.add_argument("fn_bam").required().help("Input bam.");

    p.add_group("Frequently used:");
    p.add_argument("-o").default_value(clio.output_prefix.native()).help("Output prefix.");
    p.add_argument("-t").scan<'i', int>().default_value(clio.n_threads).help("Threads to use.");
    p.add_argument("-b").flag().help("Output phased bam.");
    p.add_argument("-r", "--region")
            .help("Specify region in 0-index [) with format like chr6:10M-11000000");
    p.add_argument("-w")
            .scan<'i', int>()
            .default_value(clio.varcall_w)
            .help("Variant calling window size.");

    p.add_group("Others:");
    p.add_argument("--strict-intervals")
            .flag()
            .help("Set to disable region expansion i.e. phase using only the variants "
                  "within each chunk.")
            .action([func = __func__](const std::string &_) {
                spdlog::info("[kdys::{}] will not expand intervals during phasing.", func);
            });
    p.add_argument("--ambig-ref")
            .flag()
            .help("Set to allow the output vcf to contain")
            .action([func = __func__](const std::string &_) {
                spdlog::info(
                        "[kdys::{}] vcf output will include entries where reference allele "
                        "contains ambiguous bases",
                        func);
            });
    p.add_argument("--max-gc-seqdiv")
            .scan<'f', float>()
            .default_value(clio.pp.max_gapcompressed_seqdiv)
            .help("Max gap-compressed sequence divergence allowed for a read to "
                  "be considered in phasing & varcall.");
    p.add_argument("--max-clipping")
            .scan<'i', int>()
            .default_value(clio.pp.max_clipping)
            .help("Ignore reads with clippings larger than INT on either side.");
    p.add_argument("--min-strand-cov")
            .scan<'i', int>()
            .default_value(clio.pp.min_strand_cov)
            .help("Minimum coverage for both strands (inclusive).");
    p.add_argument("--min-strand-cov-frac")
            .scan<'f', float>()
            .default_value(clio.pp.min_strand_cov_frac)
            .help("Minimum coverage fraction for both strands (inclusive).");
    p.add_argument("-p")
            .scan<'i', int>()
            .default_value(clio.bed_flanking)
            .help("Padding size for BED output. If specified multiple times, only the last value "
                  "will be used.");
    p.add_argument("--readgroup")
            .help("Specify a readgroup to include. Default is to include all.");
    p.add_argument("--verbose").flag().help("Enable more output.");
}

void parse_cli_varcall_fill(const argparse::ArgumentParser &p, clio_pack_t &clio_pack) {
    cliopt_varcall_t &clio = clio_pack.clio_varcall;
    clio.fn_ref = std::filesystem::path(p.get("fn_ref"));
    clio.fn_bam = std::filesystem::path(p.get("fn_bam"));

    clio.output_prefix = std::filesystem::path(p.get<std::string>("-o"));
    clio.n_threads = p.get<int>("-t");
    clio.varcall_w = p.get<int>("-w");
    clio.bed_flanking = p.get<int>("-p");
    clio.pp.max_gapcompressed_seqdiv = p.get<float>("--max-gc-seqdiv");
    clio.pp.max_clipping = p.get<int>("--max-clipping");
    clio.pp.min_strand_cov = p.get<int>("--min-strand-cov");
    clio.pp.min_strand_cov_frac = p.get<float>("--min-strand-cov-frac");

    if (const auto v = p.present<std::string>("--region"); v) {  // or -r
        clio.varcall_regions.push_back(v.value());
    }
    if (const auto v = p.present<std::string>("--readgroup"); v) {
        clio.pp.readgroup = v.value();
    }
    clio.write_dbg_bam = p.get<bool>("-b");
    clio.pp.disable_region_expansion = p.get<bool>("--strict-intervals");
    clio.ambig_ref = p.get<bool>("--ambig-ref");
    kadayashi::KDY_VERBOSE = p.get<bool>("--verbose");

    throws_sancheck_bam_is_valid(clio.fn_bam);
    throws_sancheck_fastx_is_valid(clio.fn_ref);
    throws_sancheck_cliopt_varcall(clio);
}

void parse_cli_featmatgen_addarg(argparse::ArgumentParser &p, clio_pack_t &clio_pack) {
    const cliopt_featmatgen_t &clio = clio_pack.clio_featmatgen;

    p.add_argument("fn_in_ref").required().help("Input: reference genome.");
    p.add_argument("fn_in_bam").required().help("Input: BAM.");
    p.add_argument("-o").default_value(clio.fn_out.native()).help("Output file.");

    p.add_argument("-t").scan<'i', int>().default_value(clio.n_threads).help("Number of threads.");
    p.add_argument("--exclude-dwells")
            .flag()
            .help("Omit dwells from the matrix.")
            .action([func = __func__](const std::string &_) {
                spdlog::info("[kdys::{}] feature matrix will not include dwells", func);
            });
    p.add_argument("--exclude-haplotag")
            .flag()
            .help("Omit haplotags from the matrix.")
            .action([func = __func__](const std::string &_) {
                spdlog::info("[kdys::{}] feature matrix will not include haplotags", func);
            });
    p.add_argument("--exclude-snp-qv")
            .flag()
            .help("Omit SNP QVs from the matrix.")
            .action([func = __func__](const std::string &_) {
                spdlog::info("[kdys::{}] feature matrix will not include SNP QVs", func);
            });
    p.add_argument("--min-mapq")
            .scan<'i', int>()
            .default_value(clio.min_mapq)
            .help("Min MAPQ required.");
    p.add_argument("--readgroup")
            .help("Specify a readgroup to include. Default is to include all.");
    p.add_argument("--disable-read-packing")
            .flag()
            .help("Let each read reside on its own lane.")
            .action([func = __func__](const std::string &_) {
                spdlog::info(
                        "[kdys::{}] feature matrix will not pack reads. Read will appear on its "
                        "own lane.",
                        func);
            });
    p.add_argument("--max-lanes")
            .scan<'i', int>()
            .default_value(clio.max_lanes)
            .help("Cap the number of lanes at INT.");
    p.add_argument("--left-align-insertions")
            .flag()
            .help("Left align insertions (default is right-align).")
            .action([func = __func__](const std::string &_) {
                spdlog::info(
                        "[kdys::{}] feature matrix will left-align insertions. Default is "
                        "right-aligned.",
                        func);
            });
    p.add_argument("--use-medaka")
            .flag()
            .help("Use medaka's feature generator instead.")
            .action([func = __func__](const std::string &_) {
                spdlog::info(
                        "[kdys::{}] feature matrix generation will use the medaka implementation.",
                        func);
            });
    p.add_argument("--min-snp-accuracy")
            .scan<'g', double>()
            .default_value(clio.min_snp_accuracy)
            .help("Min SNP accuracy required.");
    p.add_argument("-r", "--region")
            .help("Generate feature matrix for the given region and write to file.");
    p.add_argument("--verbose").flag().help("Enable more output.");
}

void parse_cli_featmatgen_fill(const argparse::ArgumentParser &p, clio_pack_t &clio_pack) {
    cliopt_featmatgen_t &clio = clio_pack.clio_featmatgen;
    clio.fn_in_ref = std::filesystem::path(p.get("fn_in_ref"));
    clio.fn_in_bam = std::filesystem::path(p.get("fn_in_bam"));

    clio.n_threads = p.get<int>("-t");
    clio.fn_out = p.get<std::string>("-o");
    {
        clio.min_mapq = p.get<int>("--min-mapq");
        if (clio.min_mapq < 0) {
            throw std::runtime_error(
                    fmt::format("[kdys::{}] invalid value for --min-mapq. Should be a "
                                "non-negative integer.",
                                __func__));
        }
    }
    {
        clio.max_lanes = p.get<int>("--max-lanes");
        if (clio.max_lanes <= 0) {
            throw std::runtime_error(fmt::format(
                    "[kdys::{}] invalid value for --max-lanes. Should be a positive integer.",
                    __func__));
        }
    }
    {
        clio.min_snp_accuracy = p.get<double>("--min-snp-accuracy");
        if ((clio.min_snp_accuracy >= 1.0) || (clio.min_snp_accuracy < 0)) {
            throw std::runtime_error(
                    fmt::format("[kdys::{}] invalid value for --min-snp-accuracy. Should be a "
                                "double within [0.0, 1.0) .",
                                __func__));
        }
    }

    if (const auto v = p.present<std::string>("--region"); v) {  // or -r
        clio.itvl_str = v.value();
    }
    if (const auto v = p.present<std::string>("--readgroup"); v) {
        clio.readgroup = v.value();
    }
    clio.include_dwells = !p.get<bool>("--exclude-dwells");
    clio.include_haplotype_column = !p.get<bool>("--exclude-haplotag");
    clio.include_snp_qv = !p.get<bool>("--exclude-snp-qv");
    clio.disable_read_packing = p.get<bool>("--disable-read-packing");
    clio.right_align_insertions = !p.get<bool>("--left-align-insertions");
    clio.is_use_medaka = p.get<bool>("--use-medaka");
    kadayashi::KDY_VERBOSE = p.get<bool>("--verbose");
}

clio_pack_t parse_cli_main(int argc, char *argv[]) {
    // note: default values are defined in the
    // struct implementations, not here.
    clio_pack_t clio_pack;

    argparse::ArgumentParser p("kadayashi", KADAYASHI_VERSION);

    argparse::ArgumentParser p_phase("phase", KADAYASHI_VERSION);
    p_phase.add_description(
            "Given a reference genome and aligned reads in BAM, phase the reads, output read name "
            "to haptag mapping to kadayashi tsv and bin files.");

    argparse::ArgumentParser p_modifytag("modifytag", KADAYASHI_VERSION);
    p_modifytag.add_description(
            "Given BAM and mapping between read names and haptags, produce a BAM with HP:i tag "
            "added in accordingly.");

    argparse::ArgumentParser p_tsv2bin("tsv2bin", KADAYASHI_VERSION);
    p_tsv2bin.add_description("Given a kadayashi tsv file, produce a kadayashi bin file.");

    argparse::ArgumentParser p_bin2bam("bin2bam", KADAYASHI_VERSION);
    p_bin2bam.add_description(
            "Given a kadayashi bin file and a BAM, produce a BAM with HP:i tag added in "
            "accordingly.");

    argparse::ArgumentParser p_phaseglobal("phaseglobal", KADAYASHI_VERSION);
    p_phaseglobal.add_description(
            "Given a reference genome, a BAM and a vcf, produce a phased vcf.");

    argparse::ArgumentParser p_varcall("varcall", KADAYASHI_VERSION);
    p_varcall.add_description(
            "Given a reference genome and aligned reads in BAM, produce a phased vcf and some "
            "other files.");

    argparse::ArgumentParser p_featmatgen("featmatgen", KADAYASHI_VERSION);
    p_featmatgen.add_description(
            "Produce medaka-style feature matrix from reference genome and BAM.");

    parse_cli_phase_addarg(p_phase, clio_pack);
    parse_cli_modifytag_addarg(p_modifytag, clio_pack);
    parse_cli_tsv2bin_addarg(p_tsv2bin, clio_pack);
    parse_cli_bin2bam_addarg(p_bin2bam, clio_pack);
    parse_cli_phaseglobal_addarg(p_phaseglobal, clio_pack);
    parse_cli_varcall_addarg(p_varcall, clio_pack);
    parse_cli_featmatgen_addarg(p_featmatgen, clio_pack);

    p.add_subparser(p_phase);
    p.add_subparser(p_modifytag);
    p.add_subparser(p_tsv2bin);
    p.add_subparser(p_bin2bam);
    p.add_subparser(p_phaseglobal);
    p.add_subparser(p_varcall);
    p.add_subparser(p_featmatgen);

    // print help if no subcommand is given
    if (argc == 1) {
        std::ostringstream msg;
        msg << p;
        spdlog::info("{}", msg.str());
        clio_pack.is_good = false;
        return clio_pack;
    }

    // parse. if any subcommand or argument is unknown/missing it is catched here.
    try {
        p.parse_args(argc, argv);
    } catch (const std::exception &err) {
        std::ostringstream msg;
        msg << err.what();
        spdlog::error("[kdys::{}] cli parsing failed. Use -h to print help.", __func__);
        spdlog::error("{}", msg.str());
        clio_pack.is_good = false;
        return clio_pack;
    }

    // save the arguments. also does some manual sanity check or clamp.
    try {
        if (p.is_subcommand_used("phase")) {
            parse_cli_phase_fill(p_phase, clio_pack);
        } else if (p.is_subcommand_used("modifytag")) {
            parse_cli_modifytag_fill(p_modifytag, clio_pack);
        } else if (p.is_subcommand_used("tsv2bin")) {
            parse_cli_tsv2bin_fill(p_tsv2bin, clio_pack);
        } else if (p.is_subcommand_used("bin2bam")) {
            parse_cli_bin2bam_fill(p_bin2bam, clio_pack);
        } else if (p.is_subcommand_used("phaseglobal")) {
            parse_cli_phaseglobal_fill(p_phaseglobal, clio_pack);
        } else if (p.is_subcommand_used("varcall")) {
            parse_cli_varcall_fill(p_varcall, clio_pack);
        } else if (p.is_subcommand_used("featmatgen")) {
            parse_cli_featmatgen_fill(p_featmatgen, clio_pack);
        } else {
            // should not reach here
            spdlog::error("[kdys::{}] unknown subcommand", __func__);
            clio_pack.is_good = false;
        }
    } catch (const std::exception &err) {
        std::ostringstream msg;
        msg << err.what();
        spdlog::error("[kdys::{}] cli subcommand parsing failed. Use -h to print help.", __func__);
        spdlog::error("{}", msg.str());
        clio_pack.is_good = false;
    }

    return clio_pack;
}
