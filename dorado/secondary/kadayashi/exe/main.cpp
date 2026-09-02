#include "../string_utils.h"
#include "bam_tagging.h"
#include "cli.h"
#include "kadayashi_utils.h"
#include "pipeline.h"
#include "resources.h"
#include "secondary/common/bam_file.h"
#include "secondary/kadayashi/faidx_utils.h"
#include "secondary/kadayashi/local_haplotagging.h"

#include <htslib/faidx.h>
#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <cstdlib>
#include <string>
#include <string_view>

namespace {

void print_end_summary(int argc, char *argv[], double timestamp) {
    spdlog::info("\n[M::{}] ELAPSED: {:.1f}s , PeakRSS: {:.1f} GiB", __func__,
                 kadayashi::get_timestamp() - timestamp, kadayashi::get_peakrss());
    spdlog::info("[M::{}] kadayashi {}", __func__, KADAYASHI_VERSION);
    std::string cmd = "[M::";
    cmd += __func__;
    cmd += "] CMD: ";
    for (int i = 0; i < argc; i++) {
        cmd += argv[i];
        cmd += " ";
    }
    spdlog::info(cmd);
}

int main_phase(cliopt_t &clio) {
    int ret = kadayashi::local_haplotagging(clio.fn_bam, clio.fn_bed, clio.slice_in_bed,
                                            clio.fn_ref, clio.one_region_str, clio.one_chrom_str,
                                            clio.fn_vcf, clio.chunk_l, clio.chunk_stride, clio.pp,
                                            clio.n_threads,  // threads
                                            BAM_THREAD_MULT,
                                            clio.n_threads * 2,  // chunk per batch
                                            clio.fn_tsv);
    return ret;
}

int main_featmatgen(const std::filesystem::path &fn_ref,
                    const std::filesystem::path &fn_bam,
                    const std::filesystem::path &fn_out,
                    const std::string &itvl_str,  // 1-index []
                    int n_threads,
                    bool is_use_medaka_featmatgen,
                    kadayashi::MedakaFeatureMatrixOptions mfm_options) {
    // note: output will be written to file only if -r or --region is specified
    constexpr uint32_t STRIDE = 500000;

    if (!itvl_str.empty()) {
        double timestamp = kadayashi::get_timestamp();
        std::string ref_name;
        uint32_t ref_start, ref_end;  // 0-index [)
        const bool itvl_str_parse_ok =
                kadayashi::split_region_string(itvl_str, ref_name, ref_start, ref_end);
        if (!itvl_str_parse_ok) {
            spdlog::error("[kdys::{}] interval string parsing failed: {}", __func__, itvl_str);
            return 1;
        }
        dorado::secondary::BamFile hf{fn_bam, 4};
        dorado::secondary::BamFileView hf_view = hf.get_view();

        const std::unordered_map<std::string, int32_t> qname2hp{};
        kadayashi::MedakaFeatureMatrix mfm =
                is_use_medaka_featmatgen
                        ? dorado::secondary::calculate_read_alignment(
                                  hf, ref_name, ref_start, ref_end, qname2hp,
                                  mfm_options.num_dtypes, mfm_options.dtypes, mfm_options.tag_name,
                                  mfm_options.tag_value, mfm_options.tag_keep_missing,
                                  mfm_options.readgroup, mfm_options.min_mapq,
                                  mfm_options.disable_read_packing, mfm_options.include_dwells,
                                  mfm_options.include_haplotype_column, mfm_options.include_snp_qv,
                                  dorado::secondary::HaplotagSource::UNPHASED,
                                  mfm_options.max_reads, mfm_options.right_align_insertions,
                                  mfm_options.min_snp_accuracy)
                        : kadayashi::gen_medaka_feature_matrix(hf_view, ref_name, ref_start,
                                                               ref_end, qname2hp, mfm_options);
        spdlog::info("[time::{}] feature matrix generation used {:.2f}s\n", __func__,
                     kadayashi::get_timestamp() - timestamp);

        kadayashi::print_medaka_feature_matrix(fn_out, mfm);
    } else {  //  all of the bam file
        const int n_cpu_per_worker = 1;
        const int n_workers = n_threads / n_cpu_per_worker;
        double timestamp = kadayashi::get_timestamp();

        // ref
        faidx_t *fai = fai_load_format(fn_ref.string().c_str(), FAI_FASTA);

        // placeholder
        const std::unordered_map<std::string, int32_t> qname2hp{};

        // get the refseq names from the bam
        std::vector<std::string> ref_names;
        dorado::secondary::BamFile tmp_hf{fn_bam, 1};
        const int n_targets = tmp_hf.hdr()->n_targets;
        ref_names.reserve(n_targets);
        for (int i = 0; i < n_targets; i++) {
            ref_names.push_back(tmp_hf.hdr()->target_name[i]);
        }

        for (const std::string &chrom : ref_names) {
            spdlog::debug("[dbg::{}] ---- at {} ----\n", __func__, chrom.c_str());

            const std::string refseq = kadayashi::hts_utils::fetch_seq(fai, chrom);

            // (for batching)
            const uint32_t end = refseq.size();
            // (lambda)
            std::atomic<int> next_jobID{0};
            const int job_stride = 10;
            const int n_jobs = (int)end / STRIDE;
            auto worker = [&] {
                dorado::secondary::BamFile hf{fn_bam.string().c_str(), n_cpu_per_worker};
                dorado::secondary::BamFileView hf_view = hf.get_view();
                while (true) {
                    const int jobID_start =
                            next_jobID.fetch_add(job_stride, std::memory_order_relaxed);
                    if (jobID_start >= n_jobs) {
                        break;
                    }
                    for (int i_job = jobID_start; i_job < jobID_start + job_stride; i_job++) {
                        uint32_t ref_start = i_job * STRIDE;
                        uint32_t ref_end = (i_job + 1) * STRIDE;
                        kadayashi::MedakaFeatureMatrix mfm =
                                is_use_medaka_featmatgen
                                        ? dorado::secondary::calculate_read_alignment(
                                                  hf, chrom, ref_start, ref_end, qname2hp,
                                                  mfm_options.num_dtypes, mfm_options.dtypes,
                                                  mfm_options.tag_name, mfm_options.tag_value,
                                                  mfm_options.tag_keep_missing,
                                                  mfm_options.readgroup, mfm_options.min_mapq,
                                                  mfm_options.disable_read_packing,
                                                  mfm_options.include_dwells,
                                                  mfm_options.include_haplotype_column,
                                                  mfm_options.include_snp_qv,
                                                  dorado::secondary::HaplotagSource::UNPHASED,
                                                  mfm_options.max_reads,
                                                  mfm_options.right_align_insertions,
                                                  mfm_options.min_snp_accuracy)
                                        : kadayashi::gen_medaka_feature_matrix(
                                                  hf_view, chrom, ref_start, ref_end, qname2hp,
                                                  mfm_options);
                        spdlog::debug("[dbg::{}] {} {}-{} done; junk {}\n", chrom.c_str(),
                                      "main_featmatgen", ref_start, ref_end,
                                      mfm.matrix.size() > 0 ? mfm.matrix[0] : -1);
                    }
                }
            };
            // (go)
            std::vector<std::thread> threads;
            threads.reserve(n_workers);
            for (int t = 0; t < n_workers; t++) {
                threads.emplace_back(worker);
            }
            for (auto &th : threads) {
                th.join();
            }
        }
        spdlog::info("[M::{}] total time used: {:.2f}s\n", __func__,
                     kadayashi::get_timestamp() - timestamp);
    }

    return 0;
}

}  // namespace

int main(int argc, char *argv[]) {
    int ret = 0;
    double timestamp = kadayashi::get_timestamp();

    spdlog::set_level(spdlog::level::trace);

    clio_pack_t clio_pack = parse_cli_main(argc, argv);
    if (!clio_pack.is_good) {
        return 1;
    }

    const std::string_view subcommand{argv[1]};
    if (subcommand == "phase") {
        cliopt_t &clio = clio_pack.clio_phase;

        ret = main_phase(clio);

        if (ret != 0) {
            spdlog::error("[kdys::{}] phasing failed; not generating bin file from tsv output",
                          __func__);
            print_end_summary(argc, argv, timestamp);
            return ret;
        }

        std::string fn_out_binary = clio.output_prefix.string() + ".bin";
        kadayashi::write_binary_given_tsv(clio.fn_tsv, fn_out_binary);
        if (clio.write_dbg_bam) {
            std::string fn_out_bam = fn_out_binary + ".bam";
            const int writebam_err = kadayashi::write_haptagged_bam_given_bin_and_itvl(
                    clio.fn_bam, fn_out_binary, clio.one_region_str, fn_out_bam, clio.n_threads,
                    BAM_THREAD_MULT);
            if (writebam_err) {
                spdlog::error("[kdys::{}] failed to write debug haptagged bam, check code!",
                              __func__);
            } else {
                spdlog::info("[kdys::{}] wrote debug haptagged bam", __func__);
            }
        }
    } else if (subcommand == "modifytag") {
        cliopt_t &clio = clio_pack.clio_phase;
        ret = kadayashi::util_modify_tag_given_tsv(clio.fn_bam, clio.fn_tsv, clio.interval_str,
                                                   clio.n_threads, BAM_THREAD_MULT,
                                                   clio.output_prefix);
    } else if (subcommand == "tsv2bin") {
        cliopt_t &clio = clio_pack.clio_phase;
        kadayashi::write_binary_given_tsv(clio.fn_tsv, clio.output_name);
    } else if (subcommand == "bin2bam") {
        // given bam, bin and an interval, output bam with modified haplotag. Debug util.
        cliopt_t &clio = clio_pack.clio_phase;
        ret = kadayashi::write_haptagged_bam_given_bin_and_itvl(
                clio.fn_bam, clio.fn_bin, clio.one_region_str, clio.output_name, clio.n_threads,
                BAM_THREAD_MULT);
    } else if (subcommand == "phaseglobal") {
        cliopt_phaseglobal_t &clio = clio_pack.clio_phaseglobal;

        spdlog::info("[kdys::{}] start phasing with {} threads...", __func__, clio.n_threads);
        auto qname2hp = kadayashi::kadayashi_global_phasing_simple_modify_vcf(
                clio.fn_ref, clio.fn_bam, clio.fn_vcf, clio.output_name, clio.n_threads);

        if (!clio.fn_out_bam.empty()) {
            spdlog::info("[kdys::{}] To write haptagged bam...", __func__);
            double timestamp = kadayashi::get_timestamp();
            kadayashi::write_haptagged_bam_given_hashtable_and_itvl(
                    clio.fn_bam, ".", clio.fn_out_bam, qname2hp, clio.n_threads);
            spdlog::info("[kdys::{}] haptagged bam written, used {:.1f}s", __func__,
                         kadayashi::get_timestamp() - timestamp);
        }
        ret = 0;
    } else if (subcommand == "varcall") {
        cliopt_varcall_t &clio = clio_pack.clio_varcall;

        double timestamp = kadayashi::get_timestamp();
        std::string fn_out_vcf = clio.output_prefix.string() + ".vcf";
        kadayashi::str2int_t qname2hp = kadayashi::kadayashi_phased_variant_calling_threaded(
                clio.fn_ref, clio.fn_bam, clio.n_threads, fn_out_vcf, clio.output_prefix,
                clio.varcall_w, clio.varcall_regions, clio.pp.readgroup, clio.pp.min_base_quality,
                clio.pp.min_varcall_coverage, clio.pp.min_varcall_fraction, clio.pp.max_clipping,
                clio.pp.min_strand_cov, clio.pp.min_strand_cov_frac,
                clio.pp.max_gapcompressed_seqdiv, clio.ambig_ref, clio.pp.disable_region_expansion,
                clio.bed_flanking);
        spdlog::info("[kdys::{}] varcall main routine done, used {:.1f}s", __func__,
                     kadayashi::get_timestamp() - timestamp);

        if (clio.write_dbg_bam) {
            timestamp = kadayashi::get_timestamp();
            std::string fn_out_bam = clio.output_prefix.string() + ".kadayashi.bam";
            kadayashi::write_haptagged_bam_given_hashtable_and_multiple_itvls(
                    clio.fn_bam, clio.varcall_regions, fn_out_bam, qname2hp, clio.n_threads);
            spdlog::info("[kdys::{}] haptagged bam written, used {:.1f}s", __func__,
                         kadayashi::get_timestamp() - timestamp);
        }
        ret = 0;
    } else if (subcommand == "featmatgen") {
        cliopt_featmatgen_t &clio = clio_pack.clio_featmatgen;
        kadayashi::MedakaFeatureMatrixOptions mfm_options = {
                .include_dwells = clio.include_dwells,
                .include_haplotype_column = clio.include_haplotype_column,
                .include_snp_qv = clio.include_snp_qv,
                .min_mapq = clio.min_mapq,
                .num_dtypes = 1,
                .dtypes = {},
                .tag_name = "",
                .tag_value = 0,
                .tag_keep_missing = false,
                .readgroup = clio.readgroup,
                .disable_read_packing = clio.disable_read_packing,
                .hap_source = kadayashi::haplotagsource::FORCE_UNPHASED,
                .max_reads = clio.max_lanes,
                .right_align_insertions = clio.right_align_insertions,
                .min_snp_accuracy = clio.min_snp_accuracy};
        ret = main_featmatgen(clio.fn_in_ref, clio.fn_in_bam, clio.fn_out, clio.itvl_str,
                              clio.n_threads, clio.is_use_medaka, mfm_options);
    } else {
        // (should not reach here)
        spdlog::error("[E::{}] unknown subcommand: {}", __func__, subcommand);
    }

    print_end_summary(argc, argv, timestamp);
    return ret;
}
