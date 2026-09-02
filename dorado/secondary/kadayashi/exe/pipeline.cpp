#include "pipeline.h"

#include "../string_utils.h"
#include "../variant_graph.h"
#include "bam_tagging.h"
#include "hts_utils/FastxRandomReader.h"
#include "kadayashi_utils.h"
#include "resources.h"
#include "secondary/kadayashi/bam_record_parsing.h"
#include "secondary/kadayashi/sequence_utility.h"
#include "secondary/kadayashi/types.h"

#include <cxxpool.h>
#include <htslib/bgzf.h>
#include <htslib/faidx.h>
#include <htslib/hts.h>
#include <htslib/khash.h>
#include <htslib/khash_str2int.h>
#include <htslib/sam.h>
#include <spdlog/fmt/bundled/format.h>
#include <spdlog/spdlog.h>
#include <zlib.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <thread>
#include <unordered_set>

#ifdef NDEBUG
#define LOG_TRACE(...)
#else
#define LOG_TRACE(...) spdlog::trace(__VA_ARGS__)
#endif

namespace kadayashi {

namespace {

struct u32p_t {
    uint32_t s, e;
};

enum input_file_format { IS_VCF, IS_BED };

enum region_string_format { IS_WHOLE_CHROM, IS_REGULAR_SANE, IS_MALFORMAT };

region_string_format region_string_is_sane(std::string_view s, const int s_l) {
    int cnt[2] = {0, 0};  // : and -
    for (int i = 0; i < s_l; i++) {
        if (s[i] == ':') {
            cnt[0]++;
        } else if (s[i] == '-') {
            cnt[1]++;
        }
    }
    if (cnt[0] == 0 && cnt[1] == 0) {
        return IS_WHOLE_CHROM;
    }
    if (cnt[0] > 1 || cnt[1] > 1) {
        return IS_MALFORMAT;
    }
    return IS_REGULAR_SANE;
}

bool parse_region_integer(std::string_view s,
                          const size_t offset_s,
                          const size_t offset_e,
                          uint32_t *value) {
    // return false if parsing failed
    std::string tmp;
    uint32_t pos = 0;
    for (size_t i = offset_s; i < offset_e; i++) {
        if (s[i] >= 48 && s[i] <= 57) {  // 0-9
            tmp += s[i];
        } else {
            if (s[i] == ',') {
                continue;
            } else {
                uint32_t multiplier = 1;
                if (s[i] == 'g' || s[i] == 'G') {
                    multiplier = 1'000'000'000;
                } else if (s[i] == 'm' || s[i] == 'M') {
                    multiplier = 1'000'000;
                } else if (s[i] == 'k' || s[i] == 'K') {
                    multiplier = 1'000;
                } else {
                    spdlog::error("[kdys::{}] malformatted: {}", __func__, s);
                    return false;
                }

                if (tmp.empty() && multiplier > 1) {
                    spdlog::error(
                            "[kdys::{}] invalid: {} (note: we do not allow interval string like "
                            "chr1:g which should be either a chr1:1g or chr1:0)",
                            __func__, s);
                    return false;
                }

                const uint64_t num = std::stoll(tmp);
                if (num > std::numeric_limits<uint32_t>::max() / multiplier) {
                    spdlog::error("[kdys::{}] too large #1: {}", __func__, s);
                    return false;
                }

                const uint32_t tmp_pos = num * multiplier;
                if (tmp_pos > std::numeric_limits<uint32_t>::max() - pos) {
                    spdlog::error("[kdys::{}] too large #2: {}", __func__, s);
                    return false;
                }

                pos += tmp_pos;
                tmp.clear();
            }
        }
    }
    if (!tmp.empty()) {
        pos += std::stoll(tmp);
    }
    *value = pos;
    return true;
}

void insert_bed_line(const std::string &line, std::string_view chrom, std::vector<u32p_t> &intvls) {
    const int col_chrom = 0;
    const int col_s = 1;
    const int col_e = 2;
    const std::vector<std::string> cols = split_deli_line(line, '\t');
    if (cols.size() != 3) {
        spdlog::error("[kdys::{}] error parsing BED line: {}", __func__, line);
        return;
    }
    if (chrom == cols[col_chrom]) {
        uint32_t s = std::stoul(cols[col_s]);
        uint32_t e = std::stoul(cols[col_e]);
        intvls.push_back(u32p_t{.s = s, .e = e});
    }
}

int insert_variant_positions_from_a_vcf_line(const std::string &line,
                                             std::string_view chrom,
                                             const uint32_t ref_start,
                                             const uint32_t ref_end,
                                             std::vector<u32p_t> &buf) {
    // Assumes vcf is sorted. Checks whether a variant is heterozygous,
    // but will ignore phasing info (if any).
    //
    // Note: here we must NOT collect only the SNPs, even if we will only
    // use SNPs as informative sites. The reason is we will need
    // indels (1) to drop dubious calls close to homopolymers,
    // and (2) REF alleles are not collected explicitly, a position
    // might be actually covered by previous DEL rather than holding
    // the REF alelle.
    if (line.size() <= 2) {
        return -1;
    }
    int ret = 0;

    const std::vector<std::string> cols = split_deli_line(line, '\t');
    if (line[0] == '#') {
        if (line[1] == '#') {
            return -1;
        }
        if (cols.size() < 10) {
            spdlog::error(
                    "[kdys::{}] vcf only has {} columns; mandatory >=8; we also need FORMAT and "
                    "at least 1 sample. (header line check)",
                    __func__, static_cast<int>(cols.size()));
            exit(1);
        } else if (cols.size() > 10) {
            spdlog::error("[kdys::{}] multi-sample vcf not supported. (header line check)",
                          __func__);
            exit(1);
        }
        return -1;
    } else {
        if (cols.size() != 10) {
            spdlog::warn("[kdys::{}] a vcf line does not have 10 cols, was ignored: {}", __func__,
                         line);
            return -1;
        }
        if (chrom == cols[0]) {
            uint32_t pos = strtoul(cols[1].c_str(), NULL, 10);
            if (pos == 0) {
                spdlog::error(
                        "[kdys::{}] VCF line position is 0, should not happen, check input. "
                        "Offending line: {}",
                        __func__, line);
            } else {
                pos -= 1;  // convert to 0-index
            }
            if ((pos >= ref_start) && (pos < ref_end)) {
                const std::vector<std::string> tags = split_deli_line(cols[8], ':');
                const std::vector<std::string> values = split_deli_line(cols[9], ':');
                for (size_t i_tag = 0; i_tag < tags.size(); i_tag++) {
                    if (tags[i_tag] == "GT") {
                        if (values[i_tag].size() == 3 && values[i_tag][0] != values[i_tag][2]) {
                            buf.push_back(u32p_t{.s = pos, .e = pos + 1});
                            ret = 1;
                        }
                        break;
                    }
                }
            }
        }
    }

    return ret;
}

std::vector<u32p_t> load_intervals_vars_from_file_one_ref(const std::filesystem::path &fn,
                                                          std::string_view chrom,
                                                          const uint32_t ref_start,
                                                          const uint32_t ref_end,
                                                          const enum input_file_format fn_format) {
    std::vector<u32p_t> ret;
    std::vector<uint64_t> sorter;

    std::ifstream fp(fn);
    if (!fp) {
        spdlog::error("[kdys::{}] failed to open file: {}", __func__, fn.string());
        exit(1);
    }

    std::string line;
    while (std::getline(fp, line)) {
        if (fn_format == IS_BED) {
            insert_bed_line(line, chrom, ret);
            if (!ret.empty()) {
                sorter.push_back(((uint64_t)ret.back().s) << 32 | ret.back().e);
            }
        } else if (fn_format == IS_VCF) {
            insert_variant_positions_from_a_vcf_line(line, chrom, ref_start, ref_end, ret);
        }
    }
    if (fn_format == IS_BED) {  // make sure intervals loaded from bed is sorted
        std::sort(sorter.begin(), sorter.end());
        for (size_t i = 0; i < sorter.size(); i++) {
            ret[i].s = sorter[i] >> 32;
            ret[i].e = (uint32_t)sorter[i];
        }
    }

    if (fn_format == IS_VCF) {
        spdlog::info("[kdys::{}] loaded {} variant positions from vcf", __func__,
                     static_cast<int>(ret.size()));
    }

    return ret;
}

void local_haptagging_write_tsv(std::ofstream &fp,
                                const int chunkID,
                                std::string_view refname,
                                const uint32_t ref_start,
                                const uint32_t ref_end,
                                const int success,
                                const int n_reads,
                                const std::vector<std::string> &qnames,
                                const std::vector<uint8_t> &haptags,
                                const std::vector<u32p_t> &votes,
                                const std::vector<uint32_t> &informative_site_positions) {
    int n = 0;
    for (int i = 0; i < n_reads; i++) {
        if (qnames[i].empty()) {
            continue;
        }
        n++;
    }
    fp << fmt::format("C\tck.{:d}\t{:s}\t{:d}\t{:d}\t{:s}\t{:d}\n", chunkID, refname, ref_start,
                      ref_end,
                      "phased",  // unused; we now re-init as soon as phasing breaks
                      1 + n);  // +1 because it's number of lines in tsv block, not number of reads

    fp << fmt::format("V\tck.{:d}\t{:d}", chunkID, (int)informative_site_positions.size());
    for (uint32_t infopos : informative_site_positions) {
        fp << fmt::format("\t{:d}", (int)infopos);
    }
    fp << "\n";

    for (int i = 0; i < n_reads; i++) {
        if (qnames[i].empty()) {
            continue;
        }
        const int haptag = haptags[i] + 1;  // use 1-index
        fp << fmt::format("R\tck.{:d}\t{:s}\t{:d}\t{:d}\t{:d}\n", chunkID, qnames[i].c_str(),
                          haptag, haptags[i] == HAPTAG_UNPHASED ? -1 : votes[i].s,
                          haptags[i] == HAPTAG_UNPHASED ? -1 : votes[i].e);
    }
}

void local_haptagging_write_tsv2(std::ofstream &fp,
                                 dorado::secondary::BamFile &hf,
                                 const int chunkID,
                                 const std::string_view refname,
                                 const uint32_t ref_start,  // 0-index
                                 const uint32_t ref_end,    // 0-index, exclusive
                                 const str2int_t &qname2hp  // 0-index
) {
    // clang-format off
    // Note: This is for usage in varcall routine. When preparing output,
    //        unlike the earliest local haptag routines, we have a approximated
    //        global phasing and its phasing breakpoints, but have not kept
    //        the het variables used in the phasing (they are discarded before
    //        the second round of pileup - the phased varcall), nor all relevant
    //        in one & sorted struct.
    //       Thus the difference to the vanilla write_tsv is: caller will chunk
    //        by breakpoints, and here we go through the bam file to (re)collect
    //        read names for each interval, lookup the ht for their haptags,
    //        and set the following to unuseful placeholder values:
    //              - informative site positions
    //              - phasing votes of each read
    // clang-format on
    const std::string itvl = kadayashi::create_region_string(refname, ref_start, ref_end);
    HtsItrPtr bamitr =
            HtsItrPtr(sam_itr_querys(hf.idx(), hf.hdr(), itvl.c_str()), HtsItrDestructor());

    // Here we go through the bam file just to make sure
    // only relevant reads will be written to the output.
    // The haptags are supplied from qname2hp;
    // haptags in the bam file has no effect.
    str2int_t ht;
    BamPtr aln = BamPtr(bam_init1(), BamDestructor());
    while (sam_itr_next(hf.fp(), bamitr.get(), aln.get()) >= 0) {
        const char *qn = bam_get_qname(aln);
        auto it = qname2hp.find(qn);
        if (it == qname2hp.cend()) {
            ht[qn] = HAPTAG_UNPHASED;
        } else {
            ht[qn] = it->second;
        }
    }
    aln = {};

    // write chunk interval
    fp << fmt::format(
            "C\tck.{:d}\t{:s}\t{:d}\t{:d}\t{:s}\t{:d}\n", chunkID, refname, ref_start, ref_end,
            "phased",             // unused; we now re-init as soon as phasing breaks
            1 + (int)ht.size());  // +1 because it's number of lines in tsv block, not number of reads

    // placeholder: put an 0 as the lone informative position
    fp << fmt::format("V\tck.{:d}\t1\t0\n", chunkID);

    // write read tags, with two placeholder vote counts
    // (tsv and bin file uses 0-index for coordinates and 1-based for haplotags)
    for (auto &[qn, hp] : ht) {
        fp << fmt::format("R\tck.{:d}\t{:s}\t{:d}\t{:d}\t{:d}\n", chunkID, qn.c_str(), hp + 1, -1,
                          -1);
    }
}

struct worker_2a2p_pl {  // pipeline
    int n_threads;
    int n_chunks_per_batch;
    pileup_pars_t pp;
    std::filesystem::path fn_bam;  // hts_itr_query needs lock when multithreading.
                                   // easier way is to just let each thread open the
                                   // bam file. This means the input cannot be
                                   // from a pipe, but we are alreadyquerying by (possibly
                                   // overlapping) intervals anyways. To let the
                                   // impl do strictly one pass to read from pipe,
                                   // while still allowing (stride <= chunk length),
                                   // more needs to be changed.
    std::filesystem::path fn_ref;  // each thread loads its own reference faidx.
    variants_t &ht_refvars;        // if empty, use self variant pileup
    int range_i;
    int tot_offset;  // stores chunkID offset across references
    std::vector<u32p_t> ranges;
    std::string_view refname;
    std::ofstream &fp_out;
    int n_bam_threads;
};

struct worker_2a2p_st {  // step
    worker_2a2p_pl *pl;
    int chunkID_start = 0;
    int n_chunks = 0;
    std::vector<std::vector<std::string>> qnames{};          // per job buffer
    std::vector<std::vector<uint8_t>> haptags{};             // per job buffer
    std::vector<std::vector<u32p_t>> votes_diploid{};        // per job buffer
    std::vector<std::vector<uint32_t>> informative_sites{};  // per job buffer
    std::vector<uint8_t> success{};                          // fixed length
};

void local_haplotagging_callback(void *data, int job_i) {
    worker_2a2p_st *d = (worker_2a2p_st *)data;
    const int chunkID = d->chunkID_start + job_i;
    const uint32_t ref_start = d->pl->ranges[chunkID].s;
    const uint32_t ref_end = d->pl->ranges[chunkID].e;

    dorado::secondary::BamFile hf{d->pl->fn_bam, d->pl->n_bam_threads};
    dorado::hts_io::FastxRandomReader fai{d->pl->fn_ref};

    dorado::secondary::BamFileView hf_view = hf.get_view();

    chunk_t ck = variant_pileup_ht(hf_view, d->pl->ht_refvars, fai.get_raw_faidx_ptr(), nullptr,
                                   d->pl->refname, ref_start, ref_end, d->pl->pp);
    if (!ck.is_valid) {
        d->success[job_i] = 0;
        return;
    }

    int success = 0;
    const bool variant_graph_ok = variant_graph_gen(ck);
    if (variant_graph_ok) {
        const int n_iter = std::max(10, static_cast<int>(ref_end - ref_start) / 10000);
        variant_graph_do_simple_haptag(ck, n_iter);
        success = 1;
    }
    d->success[job_i] = success;

    // store reads strictly within the queried range
    d->qnames[job_i].resize(ck.reads.size());
    for (size_t i = 0; i < ck.reads.size(); i++) {  // store read info
        bool is_filler = false;
        if (ck.reads[i].end_pos < ck.abs_start || ck.reads[i].start_pos >= ck.abs_end) {
            is_filler = true;
        }
        d->haptags[job_i].push_back(
                ck.reads[i].hp);  // selfnote: pushing hp even if block is considered unphased.
        d->votes_diploid[job_i].push_back(u32p_t{.s = (uint32_t)ck.reads[i].votes_diploid[0],
                                                 .e = (uint32_t)ck.reads[i].votes_diploid[1]});
        if (is_filler) {
            d->qnames[job_i][i] = "";
        } else {
            d->qnames[job_i][i] = ck.qnames[i];
        }
    }

    // store variants
    for (const ta_t &varcall : ck.varcalls) {
        d->informative_sites[job_i].push_back(varcall.pos);
    }
}

void local_haplotagging_phase_and_write(worker_2a2p_pl *pl) {
    assert(pl->n_chunks_per_batch > 0);
    for (size_t i = 0; i < pl->ranges.size(); i += pl->n_chunks_per_batch) {
        worker_2a2p_st *st = new worker_2a2p_st();
        st->pl = pl;

        // parse bam and do phasing
        if (i >= pl->ranges.size()) {
            break;
        }
        size_t j = std::min<size_t>(pl->ranges.size(), i + pl->n_chunks_per_batch);
        if (j <= i) {
            break;
        }
        st->n_chunks = j - i;
        st->chunkID_start = pl->range_i;
        pl->range_i += st->n_chunks;

        st->qnames.resize(st->n_chunks);
        st->haptags.resize(st->n_chunks);
        st->success.resize(st->n_chunks, 0);
        st->votes_diploid.resize(st->n_chunks);
        st->informative_sites.resize(st->n_chunks);

        cxxpool::thread_pool pool{static_cast<size_t>(pl->n_threads)};
        std::vector<std::future<void>> pool_futures;
        pool_futures.reserve(st->n_chunks);
        for (int job_i = 0; job_i < st->n_chunks; job_i++) {
            pool_futures.emplace_back(pool.push(local_haplotagging_callback, st, job_i));
        }
        for (auto &v : pool_futures) {
            v.get();
        }

        // write to file
        for (int i_chunk = 0; i_chunk < st->n_chunks; i_chunk++) {
            const int chunkID_abs = st->pl->tot_offset + st->chunkID_start + i_chunk;
            const int chunkID = st->chunkID_start + i_chunk;
            const uint32_t start = pl->ranges[chunkID].s;
            const uint32_t end = pl->ranges[chunkID].e;
            local_haptagging_write_tsv(pl->fp_out, chunkID_abs, pl->refname, start, end,
                                       st->success[i_chunk], st->haptags[i_chunk].size(),
                                       st->qnames[i_chunk], st->haptags[i_chunk],
                                       st->votes_diploid[i_chunk], st->informative_sites[i_chunk]);
            spdlog::info("[kdys::{}] done processing interval {}:{}-{}", __func__, pl->refname,
                         start, end);
        }
    }
}

reference_variants_t load_frozen_variants_from_vcf_2ad(const std::filesystem::path &fn_vcf) {
    // TODO: This is a loader for 2ad phasing; need a better parser to output vcf
    //       (e.g. handle commas in alt field, handle duplicated positions).
    // note: Checks for 2-allele diploid and ignore variants that are not.

    reference_variants_t refvars;

    std::ifstream file(fn_vcf);
    if (!file.is_open()) {
        spdlog::error("[kdys::{}] failed to open input vcf: {}", __func__, fn_vcf.string());
        exit(1);
    }

    // TODO: to support gzip'd input
    unsigned char gz_magic[2] = {0, 0};
    file.read(reinterpret_cast<char *>(gz_magic), 2);
    if (gz_magic[0] == 0x1F && gz_magic[1] == 0x8B) {
        spdlog::error(
                "[kdys::{}] gz vcf input not yet supported (piping also not supported, must have a "
                "plain text file for input).",
                __func__);
        exit(1);
    }
    file.seekg(0, std::ios::beg);

    int wrote_multisample_warning = 0;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        if (line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        std::string col;
        std::string chrom;
        std::string alt_allele;
        std::string ref_allele;
        uint32_t pos;
        int i_col = 0;
        int i_GT = -1;
        int is_multi_allele = 0;  // will check whether ref or alt has comma

        while (std::getline(iss, col, '\t')) {
            if (i_col == 0) {
                chrom = col;
            } else if (i_col == 1) {
                try {
                    pos = std::stoi(col);
                } catch (const std::exception &) {
                    spdlog::error("[kdys::{}] invalid vcf line: {}", __func__, line);
                    break;
                }
                assert(pos > 0);
                pos = pos - 1;        // use 0-index
            } else if (i_col == 3) {  // REF
                ref_allele = col;
                if (ref_allele.size() == 1 && ref_allele[0] == '.') {
                    ref_allele.clear();
                }
                for (char c : ref_allele) {
                    if (c == ',') {
                        is_multi_allele = 1;
                        break;
                    }
                }
            } else if (i_col == 4) {  // ALT
                alt_allele = col;
                if (alt_allele.size() == 1 && alt_allele[0] == '.') {
                    alt_allele.clear();
                }
                for (char c : alt_allele) {
                    if (c == ',') {
                        is_multi_allele = 1;
                        break;
                    }
                }
            } else if (i_col == 8) {  // FORMAT
                std::istringstream format(col);
                std::string col_format;
                int j = 0;
                while (std::getline(format, col_format, ':')) {
                    if (col_format == "GT") {
                        i_GT = j;
                        break;
                    }
                    j++;
                }
            } else if (i_col == 9) {  //  SAMPLE
                std::istringstream sample(col);
                std::string col_sample;
                int j = 0;
                while (std::getline(sample, col_sample, ':')) {
                    if (j == i_GT && !is_multi_allele) {
                        if (col_sample.size() == 3 &&
                            (col_sample[0] == '0' || col_sample[0] == '1') &&
                            (col_sample[2] == '0' || col_sample[2] == '1') &&
                            (col_sample[0] !=
                             col_sample[2])) {  // is diploid & het, and is not multi-allele
                            uint8_t op;
                            if (ref_allele.size() == alt_allele.size()) {
                                op = VAR_OP_X;
                            } else if (ref_allele.size() > alt_allele.size()) {
                                op = VAR_OP_D;
                                if (alt_allele[0] != '.') {
                                    ref_allele.erase(0, 1);
                                }
                                alt_allele = ref_allele;  // storing the deleted string
                            } else {
                                op = VAR_OP_I;
                                if (ref_allele[0] !=
                                    '.') {  // remove ref base from the alt allele string
                                    alt_allele.erase(0, 1);
                                }
                            }
                            if (refvars[chrom].find(pos) != refvars[chrom].end()) {
                                spdlog::warn(
                                        "[kdys::{}] saw duplicated position: {}:{} (will use the "
                                        "last "
                                        "entry seen)",
                                        __func__, chrom, (int)pos);
                            }
                            refvars[chrom][pos] = {.op = op, .alt_allele = alt_allele};
                        }
                        break;
                    }
                    j++;
                }
            } else if (i_col > 9) {
                if (!wrote_multisample_warning) {
                    spdlog::warn(
                            "[kdys::{}] VCF has more than 10 columns; will only parse the first "
                            "sample present.",
                            __func__);
                    wrote_multisample_warning = 1;
                }
                break;
            }
            i_col++;
        }
    }

    int tot = 0;
    for (auto &[chrom, vars] : refvars) {
        tot += vars.size();
    }
    spdlog::info("[kdys::{}] loaded {} variants from {} references", __func__, tot,
                 (int)refvars.size());

    return refvars;
}

struct worker_simple_2a2p_st {
    chunk_t &ck;
    std::vector<uint32_t> seedreadID;
    std::vector<std::unordered_map<uint32_t, uint8_t>> arr_read2hp;  // readID to haptag hastables
};
void variant_graph_simple_haptag1_worker(void *data, int job_i) {
    worker_simple_2a2p_st *d = (worker_simple_2a2p_st *)data;
    const uint32_t seedreadID = d->seedreadID[job_i];
    d->arr_read2hp[job_i] = variant_graph_do_simple_haptag1_give_ht(d->ck, seedreadID);
    if (job_i % 500 == 1) {
        spdlog::info("[kdys::{}] iter {}/{} done...", __func__, (int)job_i,
                     (int)d->seedreadID.size());
    }
}

void variant_graph_do_simple_haptag_threaded(chunk_t &ck,
                                             const uint32_t n_iter,
                                             const int n_threads,
                                             std::unordered_map<uint32_t, uint8_t> &breakpoints) {
    constexpr bool DEBUG_PRINT = false;
    double T = kadayashi::get_timestamp();
    double T2 = T;
    worker_simple_2a2p_st st = {.ck = ck, .seedreadID = {}, .arr_read2hp = {}};

    if constexpr (DEBUG_PRINT) {
        for (const ta_t &var : ck.varcalls) {
            LOG_TRACE("[kdys::{}] info site pos {}", __func__, var.pos);
        }
    }

    // get seed readIDs
    std::vector<uint32_t> &seedreadIDs = st.seedreadID;
    const uint32_t stride = std::max<uint32_t>(1, ck.reads.size() / n_iter);

    std::unordered_set<uint32_t> knownseeds;
    std::vector<std::pair<size_t, uint32_t>> buf_readvarcnt;

    LOG_TRACE("[kdys::{}] requested {} iters", __func__, n_iter);
    for (uint32_t i_iter = 0; i_iter < n_iter; i_iter++) {
        // used the read with the most number of phasing variants within
        // the current bin
        uint32_t max_var = 0;
        uint32_t i_max_var = i_iter * stride;
        for (uint32_t j = i_iter * stride;
             j < std::min<uint32_t>((i_iter + 1) * stride, ck.reads.size()); j++) {
            uint32_t n_valid_vars = 0;
            for (qa_t &_ : ck.reads[j].vars) {
                if (ck.varcalls[_.var_idx].is_used == TA_STAT_ACCEPTED) {
                    n_valid_vars++;
                }
            }
            if (n_valid_vars > max_var) {
                max_var = n_valid_vars;
                i_max_var = j;
            }
        }

        if (max_var > 0) {
            if (seedreadIDs.size() > 0 && i_max_var == seedreadIDs.back()) {
                continue;
            }
            seedreadIDs.push_back(i_max_var);
            if constexpr (DEBUG_PRINT) {
                LOG_TRACE(
                        "[kdys::{}] collected a seed (iter#{}), qn {}, range {}:{}-{}, max_var = "
                        "{} var_size={}",
                        __func__, (int)seedreadIDs.size() - 1, ck.qnames[i_max_var], ck.refname,
                        (int)ck.reads[i_max_var].start_pos, (int)ck.reads[i_max_var].end_pos,
                        max_var, (int)ck.reads[i_max_var].vars.size());
                for (int tmpi = 0; tmpi < std::ssize(ck.reads[i_max_var].vars); tmpi++) {
                    LOG_TRACE("[kdys::{}] qn {} range {}-{}, variant#{} pos={} char={}", __func__,
                              ck.qnames[i_max_var], ck.reads[i_max_var].start_pos,
                              ck.reads[i_max_var].end_pos, tmpi, ck.reads[i_max_var].vars[tmpi].pos,
                              "ACGT_R"[ck.reads[i_max_var].vars[tmpi].allele[0]]);
                }
            }
        }
    }

    if constexpr (DEBUG_PRINT) {
        spdlog::info("[kdys::{}] collected {} seed reads (requested: {} ; used {:.1f} s)", __func__,
                     (int)st.seedreadID.size(), (int)n_iter, kadayashi::get_timestamp() - T2);
    }
    T2 = kadayashi::get_timestamp();

    // iterate
    st.arr_read2hp.resize(st.seedreadID.size());

    cxxpool::thread_pool pool{static_cast<size_t>(n_threads)};
    std::vector<std::future<void>> pool_futures;
    pool_futures.reserve(st.seedreadID.size());
    for (int job_i = 0; job_i < static_cast<int>(st.seedreadID.size()); job_i++) {
        pool_futures.emplace_back(pool.push(variant_graph_simple_haptag1_worker, &st, job_i));
    }
    for (auto &v : pool_futures) {
        v.get();
    }

    if constexpr (DEBUG_PRINT) {
        spdlog::info("[kdys::{}] all iterations done, used {:.1f} s", __func__,
                     kadayashi::get_timestamp() - T2);
    }
    T2 = kadayashi::get_timestamp();

    // do concensus and log breakpoints
    if constexpr (DEBUG_PRINT) {
        spdlog::info("[kdys::{}] normalizing...", __func__);
    }
    std::unordered_map<uint32_t, uint8_t> breakpoint_reads;
    normalize_readtaggings_ht(st.arr_read2hp, breakpoint_reads, ck);
    for (auto &[tmpreadID, _] : breakpoint_reads) {
        if (ck.reads[tmpreadID].vars.size() > 0) {
            const uint32_t pos = ck.reads[tmpreadID].vars[0].pos;
            breakpoints[pos] = 1;
            LOG_TRACE("[kdys::{}] log phaseblock break point from read {} : {}:{}", __func__,
                      ck.qnames[tmpreadID], ck.refname, pos);
        }
    }
    if constexpr (DEBUG_PRINT) {
        spdlog::info("[kdys::{}] normalized, used {:.1f} s. Haptagging reads...", __func__,
                     kadayashi::get_timestamp() - T2);
    }
    T2 = kadayashi::get_timestamp();

    std::vector<std::array<float, 3>> cnts(ck.reads.size(), {0.0f, 0.0f, 0.0f});
    for (const auto &read2hp : st.arr_read2hp) {
        for (const auto &[i_read, hp] : read2hp) {
            auto &cnt = cnts[i_read];
            if (hp == HAPTAG_UNPHASED) {
                cnt[2] += 1;
            } else {
                cnt[hp] += 1;
            }
        }
    }
    for (uint32_t i_read = 0; i_read < ck.reads.size(); i_read++) {
        const std::array<float, 3> &cnt = cnts[i_read];
        if ((cnt[0] > 3 && cnt[1] > 3 &&
             static_cast<float>(std::max(cnt[0], cnt[1])) / std::min(cnt[0], cnt[1]) < 1.5f) ||
            (cnt[0] + cnt[1] < 0.5f) || (cnt[0] == cnt[1])) {
            ck.reads[i_read].hp = HAPTAG_UNPHASED;
        } else {
            if (cnt[0] > cnt[1]) {
                ck.reads[i_read].hp = 0;
            } else {
                ck.reads[i_read].hp = 1;
            }
        }

        LOG_TRACE("[kdys::{}] qn {} hp {}; cnt: {:.1f} {:.1f} {:.1f}", __func__, ck.qnames[i_read],
                  ck.reads[i_read].hp, cnt[0], cnt[1], cnt[2]);
    }

    spdlog::info("[kdys::{}] reads tagged, used {:.1f} s", __func__,
                 kadayashi::get_timestamp() - T2);
    spdlog::info("[kdys::{}] haptag callback all done, used  {:.1f} s", __func__,
                 kadayashi::get_timestamp() - T);
}

chunk_t kadayashi_global_phasing_simple1(dorado::secondary::BamFileView &hf_view,
                                         const faidx_t *fai_view,
                                         std::string_view refname,
                                         const uint32_t ref_len,
                                         const variants_t &variants,  // pos and the alt allele
                                         std::unordered_map<uint32_t, uint8_t> &breakpoints,
                                         const int n_threads,
                                         const pileup_pars_t &pp) {
    // Phase one chromosome.
    double T = kadayashi::get_timestamp();

    // parse bam and collect variants on the reads
    spdlog::info("[kdys::{}] pileup... (ref {}, len {})", __func__, refname, (int)ref_len);
    chunk_t ck = variant_pileup_ht(hf_view, variants, fai_view, nullptr, refname, 1, ref_len, pp);
    spdlog::info("[kdys::{}] pileup done, has {} variants, used {:.1f} s", __func__,
                 (int)ck.varcalls.size(), kadayashi::get_timestamp() - T);

    // phase
    if (ck.is_valid) {
        const int n_iter = std::max(10, static_cast<int>(ck.abs_end - ck.abs_start) / 10000);

        const bool variant_graph_ok = variant_graph_gen(ck);
        if (variant_graph_ok) {
            spdlog::info("[kdys::{}] phasing requested {} iterations (ref length {} bp)", __func__,
                         n_iter, (int)(ck.abs_end - ck.abs_start));
            variant_graph_do_simple_haptag_threaded(ck, n_iter, n_threads, breakpoints);

            int n_reads = 0;
            int n_haps[2] = {0, 0};
            for (const read_t &read : ck.reads) {
                if (read.hp != HAPTAG_UNPHASED) {
                    n_haps[read.hp] += 1;
                }
                n_reads += 1;
            }
            spdlog::info("[kdys::{}] total of {} reads, hap0 {}, hap1 {}", __func__, n_reads,
                         n_haps[0], n_haps[1]);
        }
    } else {
        spdlog::warn("[kdys::{}] pileup failed", __func__);
    }
    return ck;
}

void haptag_variants_2ad(dorado::secondary::BamFile &hf,
                         const char *refname,
                         const variants_t &vars,  // known variants of the current chromosome
                         const str2int_t &qname2hp,
                         var2hap_t &var2hap) {
    constexpr bool DEBUG_PRINT = false;

    std::vector<std::array<int, 4>> counter(vars.size(), {0, 0, 0, 0});
    // ^stores haptag of ref allele; hap0, hap1, unphased, allele not matched

    HtsItrPtr bamitr = HtsItrPtr(sam_itr_querys(hf.idx(), hf.hdr(), refname), HtsItrDestructor());
    BamPtr aln = BamPtr(bam_init1(), BamDestructor());  // use local buffer instead of hf->aln

    // index known locations
    std::unordered_map<uint32_t, int> pos2idx;  // position-on-ref to index-in-linear-buffer
    std::unordered_map<uint32_t, int>
            pos2idx_binned;      // index, but use round numbers rather than variant positions
    std::vector<uint32_t> poss;  // all valid positions, sorted

    const int bin_size = 1000;

    for (auto &pair : vars) {
        poss.push_back(pair.first);
    }
    std::sort(poss.begin(), poss.end());
    for (size_t i = 0; i < poss.size(); i++) {
        pos2idx[poss[i]] = i;
    }

    uint32_t bottom = 0;
    for (uint32_t pos : poss) {
        uint32_t current = pos / bin_size;
        if (current > bottom) {
            for (uint32_t i = bottom; i < current; i++) {
                pos2idx_binned[i] = poss.size() == 1 ? 0 : poss.size() - 1;
            }
            pos2idx_binned[current] = poss.size() - 1;
            bottom = current;
        }
    }

    uint64_t n_reads = 0;
    std::vector<qa_t> tmp_qav;
    int base_q_min = 5;
    int left_clip_len = 0;
    int right_clip_len = 0;
    std::unordered_map<uint32_t, uint8_t> seen;
    while (sam_itr_next(hf.fp(), bamitr.get(), aln.get()) >= 0) {
        n_reads++;
        char *qn = bam_get_qname(aln.get());

        // only parse reads that were used for phasing
        auto it = qname2hp.find(qn);
        if (it == qname2hp.cend()) {
            continue;
        }
        uint8_t hp_raw = it->second;
        uint32_t start_pos = aln.get()->core.pos;
        uint32_t end_pos = bam_endpos(aln.get());

        // parse alt alleles of the read
        seen.clear();
        tmp_qav.clear();
        const bool parse_ok = parse_variants_for_one_read(aln.get(), tmp_qav, base_q_min,
                                                          &left_clip_len, &right_clip_len, 1, NULL);

        if (!parse_ok) {
            spdlog::error(
                    "[kdys::{}] read parse failed but we have had the read (qn={}), check code?",
                    __func__, qn);
            exit(1);
        } else {
            LOG_TRACE("[kdys::{}] qn {} (hp {})", __func__, qn, hp_raw);
            std::stable_sort(tmp_qav.begin(), tmp_qav.end());
            for (qa_t &q : tmp_qav) {
                uint32_t qpos = q.pos;

                // check position
                if constexpr (DEBUG_PRINT) {
                    LOG_TRACE("kdys::[{}]   pos={}", __func__, qpos);
                }
                auto it_vars = vars.find(qpos);
                if (it_vars == vars.cend()) {
                    continue;
                }
                seen[qpos] = 1;

                // check cigar operation
                const variant_t &t = it_vars->second;
                if (q.allele.back() != t.op) {
                    if constexpr (DEBUG_PRINT) {
                        LOG_TRACE("[kdys::{}]   ^ failed cigar op check (q:{} t:{})", __func__,
                                  q.allele.back(), t.op);
                    }
                    continue;
                }

                // self is alt allele, check if sequence mathces
                const std::string alt = nt4seq2seq(q.allele);  // last slot is cigar op
                if (alt != t.alt_allele) {
                    if constexpr (DEBUG_PRINT) {
                        LOG_TRACE("[kdys::{}]   failed alt allele check (q:{} t:{})", __func__, alt,
                                  t.alt_allele);
                    }

                    // this counter is neede for indels and sv when we do not have proper consensus
                    counter[pos2idx[qpos]][3] += 1;

                    continue;
                }

                if (hp_raw == HAPTAG_UNPHASED) {
                    counter[pos2idx[qpos]][2] += 1;
                } else {
                    counter[pos2idx[qpos]][hp_raw ^ 1] += 1;
                }

                if constexpr (DEBUG_PRINT) {
                    LOG_TRACE("[kdys::{}]   ^ok", __func__);
                }
            }
        }

        // recover ref alleles of the read
        uint32_t tmpi = start_pos / bin_size;
        const uint32_t idx_start =
                pos2idx_binned.find(tmpi) == pos2idx_binned.end() ? 0 : pos2idx_binned[tmpi];
        tmpi = end_pos / bin_size - 1;
        const uint32_t idx_end = pos2idx_binned.find(tmpi) == pos2idx_binned.end()
                                         ? poss.size()
                                         : pos2idx_binned[tmpi];
        for (uint32_t i = idx_start; i < idx_end; i++) {
            const uint32_t pos = poss[i];
            if (seen.find(pos) != seen.end()) {  // is an alt allele and already counted
                continue;
            }
            if constexpr (DEBUG_PRINT) {
                LOG_TRACE("[kdys::{}] read allele at pos {} (hp: {})", __func__, pos, hp_raw);
            }
            if (hp_raw == HAPTAG_UNPHASED) {
                counter[pos2idx[pos]][2] += 1;
            } else {
                counter[pos2idx[pos]][hp_raw] += 1;
            }
        }
        if constexpr (DEBUG_PRINT) {
            LOG_TRACE("[kdys::{}] end of qn {}", __func__, qn);
        }
    }

    for (size_t i = 0; i < counter.size(); i++) {
        const uint32_t pos = poss[i];
        uint8_t hp = HAPTAG_UNPHASED;
        if constexpr (DEBUG_PRINT) {
            LOG_TRACE(
                    "[kdys::{}] pos={} counter: hap0={}, hap1={}, unphased={}, "
                    "unmatched_allele={}",
                    __func__, (int)pos, counter[i][0], counter[i][1], counter[i][2], counter[i][3]);
        }

        // clang-format off
        if ((counter[i][0]>=3 && counter[i][1]>=3)  // ambiguous
            || (counter[i][0] < 3 && counter[i][1] < 3)  // both low coverage
            || (counter[i][3] >= std::max(counter[i][0], counter[i][1]))  // too many unmatched alt allele
        ) {
            hp = HAPTAG_UNPHASED;
        } else if (counter[i][0] > counter[i][1]) {
            hp = 0;
        } else if (counter[i][0] < counter[i][1]) {
            hp = 1;
        }
        var2hap[pos] = hp;
        // clang-format on
    }
}

void vcfio_alter_phasings(
        varhaps_t &varhaps,
        const std::filesystem::path &fn_vcf,
        const std::filesystem::path &fn_out_vcf,
        const std::unordered_map<std::string, std::unordered_map<uint32_t, uint8_t>>
                &phase_breakpoints) {
    constexpr bool DEBUG_PRINT = false;
    double T = kadayashi::get_timestamp();

    std::ifstream fp_in(fn_vcf);
    if (!fp_in.is_open()) {
        spdlog::error("[kdys::{}] failed to open input vcf when trying to write output", __func__);
        exit(1);
    }

    std::ofstream fp_out(fn_out_vcf);
    if (!fp_out.is_open()) {
        spdlog::error("[kdys::{}] failed to open output file: {}", __func__, fn_out_vcf.string());
        exit(1);
    }

    std::string line;
    std::vector<std::string> newline;
    int is_unmodified = 0;

    int phaseblockID = -1;
    int PS_is_defined_in_header = 0;
    std::string prev_chrom = "";
    while (std::getline(fp_in, line)) {
        if (line.size() < 2) {
            spdlog::warn("[kdys::{}] saw abnormally short vcf line: {}", __func__, line);
            continue;
        }
        if (line[0] == '#') {
            // test if PS tag has been defined
            if (line[1] == '#' && line.size() >= 16 && line.substr(0, 16) == "##FORMAT=<ID=PS,") {
                PS_is_defined_in_header = 1;
            }

            // If we reached the header line and PS tag hasn't been defined,
            // add it now.
            if (line[1] != '#' && !PS_is_defined_in_header) {
                fp_out << "##FORMAT=<ID=PS,Number=1,Type=Integer,Description=\"ID of Phase Set for "
                          "Variant\">\n";
            }

            fp_out << line << "\n";
            continue;
        }
        newline.clear();
        is_unmodified = 0;

        std::istringstream iss(line);
        std::string col;
        std::string chrom;
        std::string alt_allele;
        std::string ref_allele;
        uint32_t pos;
        int i_col = 0;
        int i_GT = -1;
        int i_PS = -1;
        uint8_t hp = HAPTAG_UNPHASED;
        int do_check_breakpoint = 0;

        while (std::getline(iss, col, '\t')) {
            if (i_col == 0) {
                chrom = col;
                if (chrom != prev_chrom) {
                    phaseblockID = -1;
                    prev_chrom = chrom;
                }
                if (varhaps.find(chrom) == varhaps.end()) {
                    is_unmodified = 1;
                    break;
                }
                auto it_pb = phase_breakpoints.find(chrom);
                if (it_pb != phase_breakpoints.cend()) {
                    do_check_breakpoint = 1;
                } else {
                    do_check_breakpoint = 0;
                }
                hp = HAPTAG_UNPHASED;
            } else if (i_col == 1) {
                try {
                    pos = std::stoi(col);
                } catch (const std::exception &) {
                    spdlog::error("[kdys::{}] invalid vcf line: {}", __func__, line);
                    is_unmodified = 1;
                    break;
                }
                assert(pos > 0);
                pos -= 1;

                int prev_is_phase_gap = 0;
                if (do_check_breakpoint) {
                    auto it_pb = phase_breakpoints.find(chrom);
                    if (it_pb->second.find(pos) != it_pb->second.end()) {
                        prev_is_phase_gap = 1;
                    }
                }

                var2hap_t &ref2hap = varhaps[chrom];
                if (ref2hap.find(pos) != ref2hap.end()) {
                    if (ref2hap[pos] != HAPTAG_UNPHASED) {
                        hp = ref2hap[pos];
                        if constexpr (DEBUG_PRINT) {
                            LOG_TRACE("[kdys::{}] pos {}, phased as hp {}", __func__, pos, hp);
                        }
                        if (phaseblockID < 0 || prev_is_phase_gap) {
                            phaseblockID = pos + 1;  // use 1-index
                            if constexpr (DEBUG_PRINT) {
                                LOG_TRACE("[kdys::{}] update phaseblock ID at {}:{} (0-index)",
                                          __func__, chrom, pos);
                            }
                        }
                    } else {
                        if constexpr (DEBUG_PRINT) {
                            LOG_TRACE("[kdys::{}] pos {}, in record but is unphased", __func__,
                                      pos);
                        }
                    }
                } else {
                    if constexpr (DEBUG_PRINT) {
                        LOG_TRACE("[kdys::{}] pos {}, not found", __func__, pos);
                    }
                    hp = HAPTAG_UNPHASED;
                }
            } else if (i_col == 3) {  // REF
                ref_allele = col;
                if (ref_allele.size() == 1 && ref_allele[0] == '.') {
                    ref_allele.clear();
                }
                for (char c : ref_allele) {
                    if (c == ',') {  // is multiallele
                        hp = HAPTAG_UNPHASED;
                        break;
                    }
                }
            } else if (i_col == 4) {  // ALT
                alt_allele = col;
                if (alt_allele.size() == 1 && alt_allele[0] == '.') {
                    alt_allele.clear();
                }
                for (char c : alt_allele) {
                    if (c == ',') {  // is multiallele
                        hp = HAPTAG_UNPHASED;
                        break;
                    }
                }
            } else if (i_col == 8) {  // FORMAT
                std::istringstream format(col);
                std::string col_format;
                int j = 0;
                while (std::getline(format, col_format, ':')) {
                    if (col_format == "GT") {
                        i_GT = j;
                    } else if (col_format == "PS") {
                        i_PS = j;
                    }
                    j++;
                }
            } else if (i_col == 9) {  //  SAMPLE
                std::istringstream sample(col);
                std::string col_sample;
                std::string &col_sample_new = newline.emplace_back("");
                int j = 0;
                while (std::getline(sample, col_sample, ':')) {
                    if (j == i_GT) {
                        if (col_sample.size() == 3 &&
                            (col_sample[0] == '0' || col_sample[0] == '1') &&
                            (col_sample[2] == '0' ||
                             col_sample[2] == '1')) {  // is diploid, and has 2 alleles

                            if (hp != HAPTAG_UNPHASED || phaseblockID >= 0) {
                                // new genotype
                                if (col_sample_new.size() != 0) {
                                    col_sample_new += ":";
                                }
                                if (hp == 0) {
                                    if (col_sample[0] == col_sample[2]) {
                                        if (col_sample[0] == '0') {
                                            col_sample_new += "0/0";
                                        } else {
                                            col_sample_new += "1/1";
                                        }
                                    } else {
                                        col_sample_new += "0|1";
                                    }
                                } else if (hp == 1) {
                                    if (col_sample[0] == col_sample[2]) {
                                        if (col_sample[0] == '0') {
                                            col_sample_new += "1|1";
                                        } else {
                                            col_sample_new += "0|0";
                                        }
                                    } else {
                                        col_sample_new += "1|0";
                                    }
                                } else {
                                    col_sample_new += col_sample[0];
                                    col_sample_new += "/";
                                    col_sample_new += col_sample[2];
                                }
                            } else {
                                if (col_sample_new.size() != 0) {
                                    col_sample_new += ":";
                                }
                                col_sample_new += std::string(1, col_sample[0]);
                                col_sample_new += "/";
                                col_sample_new += std::string(1, col_sample[2]);
                            }
                        } else {  // no change to the GT field
                            col_sample_new += col_sample;
                        }
                    } else if (j == i_PS) {
                        if (col_sample_new.size() != 0) {
                            col_sample_new += ":";
                        }
                        if (hp == HAPTAG_UNPHASED) {
                            col_sample_new += ".";
                        } else {
                            col_sample_new += std::to_string(phaseblockID);
                        }
                    } else {
                        if (col_sample_new.size() != 0) {
                            col_sample_new += ":";
                        }
                        col_sample_new += col_sample;
                    }
                    j++;
                }
                if (i_PS < 0) {  // input doesn't have phaseblockID field in col 8, so
                    // col 8 and 9 will have it appened as the last entry in their values
                    if (col_sample_new.size() != 0) {
                        col_sample_new += ":";
                    }
                    if (hp == HAPTAG_UNPHASED) {
                        col_sample_new += ".";
                    } else {
                        col_sample_new += std::to_string(phaseblockID);
                    }
                }
            } else if (i_col > 9) {
                spdlog::error("[kdys::{}] multi-sample vcf not supported", __func__);
                exit(1);
            }

            if (is_unmodified) {  // invalid line or line has multi alleles (not supported; TODO)
                break;
            }
            if (i_col < 9) {
                newline.push_back(col);
                if (i_col == 8 && i_PS < 0) {
                    newline.back() += ":PS";
                }
            }
            i_col++;
        }  // parse of one line

        if (is_unmodified) {
            fp_out << line << "\n";
        } else {
            for (size_t i = 0; i < newline.size() - 1; i++) {
                fp_out << newline[i];
                fp_out << "\t";
            }
            fp_out << newline[newline.size() - 1];
            fp_out << "\n";
        }
    }

    fp_in.close();
    fp_out.close();

    spdlog::info("[kdys::{}] written output vcf, used {:.1f} s", __func__,
                 kadayashi::get_timestamp() - T);
}

std::unordered_map<std::string, int> kadayashi_global_phasing_simple_modify_vcf1(
        reference_variants_t *refvars,
        const std::filesystem::path &fn_ref,
        const std::filesystem::path &fn_bam,
        const std::filesystem::path &fn_in_vcf,   // optional
        const std::filesystem::path &fn_out_vcf,  // optional
        const int n_threads) {
    double T = kadayashi::get_timestamp();

    if (!fn_in_vcf.native().empty() && fn_out_vcf.native().empty()) {
        spdlog::error("[kdys::{}] VCF input was provided, but did not specify output VCF name",
                      __func__);
        exit(1);
    }

    dorado::secondary::BamFile hf{fn_bam, n_threads};
    dorado::hts_io::FastxRandomReader fp_fai{fn_ref};

    dorado::secondary::BamFileView hf_view = hf.get_view();

    varhaps_t varhaps;
    std::unordered_map<std::string, int> qname2hp;
    std::unordered_map<std::string, std::unordered_map<uint32_t, uint8_t>> phase_breakpoints;
    for (auto &[chrom, vars] : (*refvars)) {
        spdlog::info("[kdys::{}] phasing {}...", __func__, chrom);
        const int ref_len = fp_fai.fetch_seq_len(chrom);

        if (ref_len < 0) {
            continue;
        }

        pileup_pars_t pp = {
                .disable_region_expansion = true,
        };

        chunk_t ck = kadayashi_global_phasing_simple1(hf_view, fp_fai.get_raw_faidx_ptr(), chrom,
                                                      (uint32_t)ref_len, vars,
                                                      phase_breakpoints[chrom], n_threads, pp);
        str2int_t tmp_qname2hp = kadayashi_local_haptagging_gen_ht(ck);
        for (auto &[qn, hp] : tmp_qname2hp) {
            if (qname2hp.find(qn) == qname2hp.end()) {
                qname2hp[qn] = hp;  // 0-index
            }
        }

        haptag_variants_2ad(hf, chrom.c_str(), vars, tmp_qname2hp, varhaps[chrom]);

        int counter[3] = {0, 0, 0};
        for (auto &[pos, hp] : varhaps[chrom]) {
            if (hp == 0) {
                counter[0]++;
            } else if (hp == 1) {
                counter[1]++;
            } else if (hp == HAPTAG_UNPHASED) {
                counter[2]++;
            }
        }
        LOG_TRACE("[kdys::{}] {} variants: hap0={} hap1={} unphased={}", __func__, chrom,
                  counter[0], counter[1], counter[2]);
    }

    // optional: alter vcf
    if (!fn_in_vcf.native().empty()) {
        vcfio_alter_phasings(varhaps, fn_in_vcf, fn_out_vcf, phase_breakpoints);
    }
    spdlog::info("[kdys::{}] used {:.1f} s", __func__, kadayashi::get_timestamp() - T);
    return qname2hp;
}

query_regions_t region_strings_to_ht(const std::vector<std::string> &query_regions) {
    query_regions_t ret0;
    query_regions_t ret;
    if (query_regions.empty()) {
        return ret;
    }

    for (const auto &s : query_regions) {
        if (s == ".") {
            ret0.clear();
            return ret0;
        }
        const region_string_format stat = region_string_is_sane(s, s.size());
        if (stat == IS_WHOLE_CHROM) {
            ret0[s].push_back(
                    {.chrom = s.c_str(), .start = 0, .end = 0});  // sentinel for whole chrom
        } else if (stat == IS_REGULAR_SANE) {
            region_string_t region = parse_region_string2(s);
            if (!region.is_parse_success) {
                spdlog::error("[kdys::{}] failed to parse the query reigon string: {}", __func__,
                              s);
            } else {
                ret0[region.chrom].push_back(
                        {.chrom = region.chrom, .start = region.start, .end = region.end});
            }
        } else {
            spdlog::error("[kdys::{}] failed to parse the query region string: {}", __func__, s);
        }
    }

    // ensure ordering
    for (auto &[chrom, d] : ret0) {
        ret[chrom] = {};
        std::sort(d.begin(), d.end(), [](const query_region_t &a, const query_region_t &b) {
            return a.start != b.start ? a.start < b.start : a.end < b.end;
        });
    }

    // ensure unique and no overlap
    for (const auto &[chrom, d] : ret0) {
        if (d.size() == 0) {
            continue;
        }
        if (d[0].start == 0 && d[0].end == 0) {
            continue;
        }
        for (auto &_ : ret0[chrom]) {
            if (ret[chrom].size() > 0 && ret[chrom].back().end >= _.start) {
                if (ret[chrom].back().end < _.end) {
                    ret[chrom].back().end = _.end;
                }
            } else {
                ret[chrom].push_back(_);
            }
        }
    }

    return ret;
}

void varcall_write_simple_vcf_header(std::ofstream &fp_out_vcf, dorado::secondary::BamFile &hf) {
    fp_out_vcf << "##fileformat=VCFv4.2\n";
    fp_out_vcf << "##FILTER=<ID=PASS,Description=\"called\">\n";
    fp_out_vcf << "##FILTER=<ID=unsr,Description=\"go to the large model\">\n";
    for (auto i = 0; i < hf.hdr()->n_targets; i++) {
        fp_out_vcf << fmt::format("##contig=<ID={:s},length={:d}>\n", hf.hdr()->target_name[i],
                                  hf.hdr()->target_len[i]);
    }
    fp_out_vcf << "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n";
    fp_out_vcf << "##FORMAT=<ID=PS,Number=1,Type=String,Description=\"PhaseblockID\">\n";
    fp_out_vcf << "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n";
}

struct varcall_chunk_phasing_t {
    str2int_t qname2hp;
    std::unordered_map<uint32_t, uint8_t> breakpoints;
    str2int_t last_names;
    str2int_t first_names;
};

enum get_phased_read_qname2hp_which_t { FRONT, BACK };

str2int_t get_phased_read_qnames2hp(
        const chunk_t &ck,
        const std::unordered_map<uint32_t, uint8_t> &phasing_breakpoints,
        const size_t max_n,
        const get_phased_read_qname2hp_which_t which_end) {
    str2int_t ht;
    const size_t n_reads = ck.reads.size();
    if (!ck.reads.empty()) {
        uint32_t first_breakpoint_pos = std::numeric_limits<uint32_t>::max();
        uint32_t last_breakpoint_pos = 0;
        for (auto &[pos, _] : phasing_breakpoints) {
            if (pos < first_breakpoint_pos) {
                first_breakpoint_pos = pos;
            }
            if (pos > last_breakpoint_pos) {
                last_breakpoint_pos = pos;
            }
        }

        const int min_i = (int)(max_n > n_reads ? 0 : n_reads - max_n);
        if (which_end == FRONT) {
            for (size_t i = 0; i < std::min<size_t>(max_n, ck.reads.size()); i++) {
                if (ck.reads[i].start_pos < first_breakpoint_pos) {
                    ht[ck.qnames[i]] = ck.reads[i].hp;
                }
            }
        } else if (which_end == BACK) {
            for (int64_t i = std::ssize(ck.reads) - 1; i >= min_i; i--) {
                if (ck.reads[i].start_pos > last_breakpoint_pos) {
                    ht[ck.qnames[i]] = ck.reads[i].hp;
                } else {
                    break;
                }
            }
        }
    }
    return ht;
}

std::string make_vcf_line_given_variant_fullinfo_t(std::string_view ref_name,
                                                   const uint32_t ref_start,
                                                   const uint32_t ref_end,
                                                   const std::vector<uint32_t> &breakpoints_arr,
                                                   const variant_fullinfo_t &var,
                                                   uint32_t &phaseblockID_fallback) {
    // note:
    //   - need to first calcualte phaseblockID, then format the line.
    //   - multi-allelic variant must either be one line whose genotype
    //     contains 1 and 2, or two lines and their genotypes contain 0 and 1.
    //     Using the latter representation here.
    if (!var.is_valid) {
        return {};
    }
    std::string oline0_s;
    std::string oline1_s;

    char genotype0[3];
    char genotype1[3];
    std::memcpy(genotype0, var.genotype0, 3);
    std::memcpy(genotype1, var.genotype1, 3);

    if (var.is_multi_allele) {
        genotype0[0]--;
        genotype0[2]--;
        genotype1[0]--;
        genotype1[2]--;
    }

    // decide phaseblock ID
    int phaseblockID = -1;
    const bool a1_is_phased_het = (genotype0[0] != genotype0[2]) && var.is_phased0;
    const bool a2_is_phased_het = (genotype1[0] != genotype1[2]) && var.is_phased1;
    assert(!a1_is_phased_het || (genotype0[1] == '|'));
    assert(!a2_is_phased_het || (genotype1[1] == '|'));
    if (a1_is_phased_het || a2_is_phased_het) {
        if (phaseblockID_fallback == 0) {
            phaseblockID_fallback = var.pos0 + 1;  // use 1-index
        }
        const auto it =
                std::upper_bound(breakpoints_arr.cbegin(), breakpoints_arr.cend(), var.pos0);
        if (it != breakpoints_arr.cbegin()) {
            phaseblockID = (*(it - 1)) + 1;  // use 1-index
        } else {
            if (!breakpoints_arr.empty() && (breakpoints_arr[0] <= var.pos0)) {
                phaseblockID = breakpoints_arr[0] + 1;  // use 1-index
            } else {
                phaseblockID = phaseblockID_fallback;
            }
        }
    }

    // REF and ALT
    oline0_s = std::string(ref_name) + "\t" + std::to_string(var.pos0 + 1) + "\t.\t";
    oline0_s += var.ref_allele_seq0 + "\t" + var.alt_allele_seq0;
    if (var.is_multi_allele) {
        oline1_s = std::string(ref_name) + "\t" + std::to_string(var.pos1 + 1) + "\t.\t";
        oline1_s += var.ref_allele_seq1 + "\t" + var.alt_allele_seq1;
    }

    // qual, filter, info, format
    oline0_s += "\t44\t";
    oline0_s += (var.is_confident ? "PASS" : "unsr");
    oline0_s += "\t.\tGT:PS";
    if (var.is_multi_allele) {
        oline1_s += "\t44\t";
        oline1_s += (var.is_confident ? "PASS" : "unsr");
        oline1_s += "\t.\tGT:PS";
    }

    // sample
    oline0_s += "\t";
    oline0_s.append(genotype0, 3);
    oline0_s += ":";
    if (var.is_multi_allele) {  // the second allele on different line
        oline1_s += "\t";
        oline1_s.append(genotype1, 3);
        oline1_s += ":";
    }

    if (!var.is_confident || !var.is_phased0 || phaseblockID < 0) {
        oline0_s += ".";
        oline1_s += ".";
    } else {
        oline0_s += std::to_string(phaseblockID);
        oline1_s += std::to_string(phaseblockID);
    }
    oline0_s += "\n";
    oline1_s += "\n";

    std::string ret;
    if (var.is_multi_allele) {
        if (var.pos0 <= var.pos1) {
            ret = oline0_s;
            ret += oline1_s;
        } else {
            ret = oline1_s;
            ret += oline0_s;
        }
    } else {
        ret = oline0_s;
    }
    return ret;
}

struct varcall_result_and_localphasinght_t {
    varcall_result_internal_t vr;
    std::unordered_map<std::string, int> qname2hp_local;
};
std::vector<varcall_result_and_localphasinght_t> kadayashi_phase_and_varcall_multiregionthreaded(
        const std::filesystem::path &fn_ref,
        const std::filesystem::path &fn_bam,
        const int n_workers,
        const int n_bam_threads,
        const std::string_view ref_name,
        const std::vector<std::pair<uint32_t, uint32_t>> &query_intervals,
        const std::string_view readgroup,
        const bool disable_interval_expansion,
        const int min_base_quality,
        const int min_varcall_coverage,
        const float min_varcall_fraction,
        const int max_clipping,
        const int min_strand_cov,
        const float min_strand_cov_frac,
        const float max_gapcompressed_seqdiv,
        const bool ambig_ref) {
    // Note: `n_workers` is the apprent # of workers; each worker's
    //       bam parsing will use n_bam_threads (>=1).
    //       Thus the total threads used is n_workers*n_bam_threads.
    if (query_intervals.empty()) {
        return {};
    }

    constexpr int N_REF_READS = 100;

    const int n_jobs = query_intervals.size();
    constexpr int JOB_CHUNK_SIZE = 1;
    std::atomic<int> next_jobID{0};

    // do variant calling in queries
    std::vector<ck_and_varcall_result_t> ck_and_vrs(n_jobs);
    auto worker = [&] {
        dorado::secondary::BamFile hf{fn_bam, n_bam_threads};
        dorado::hts_io::FastxRandomReader fp_fai{fn_ref};
        while (true) {
            const int jobID_start = next_jobID.fetch_add(JOB_CHUNK_SIZE, std::memory_order_relaxed);
            if (jobID_start >= n_jobs) {
                break;
            }

            LOG_TRACE("[kdys::{}] worker got jobID starting from {}", __func__, (int)jobID_start);

            const int jobID_end = std::min<int>(jobID_start + JOB_CHUNK_SIZE, n_jobs);
            for (int jobID = jobID_start; jobID < jobID_end; jobID++) {
                const uint32_t ref_start = query_intervals[jobID].first;
                const uint32_t ref_end = query_intervals[jobID].second;
                ck_and_varcall_result_t tmp = kadayashi_phase_and_varcall(
                        hf.fp(), hf.idx(), hf.hdr(), fp_fai.get_raw_faidx_ptr(), ref_name,
                        ref_start, ref_end, readgroup, disable_interval_expansion, min_base_quality,
                        min_varcall_coverage, min_varcall_fraction, max_clipping, min_strand_cov,
                        min_strand_cov_frac, max_gapcompressed_seqdiv, ambig_ref);
                ck_and_vrs[jobID] = std::move(tmp);
            }
        }
    };
    std::vector<std::thread> threads;
    threads.reserve(n_workers);
    for (int t = 0; t < n_workers; t++) {
        threads.emplace_back(worker);
    }
    for (auto &th : threads) {
        th.join();
    }

    // construct the return
    std::vector<varcall_result_and_localphasinght_t> ret{static_cast<std::size_t>(n_jobs)};

    // save local phasings
    for (int i = 0; i < n_jobs; i++) {
        for (const auto &[qn, hp] : ck_and_vrs[i].vr.qname2hp) {
            ret[i].qname2hp_local[qn] = hp;
        }
    }

    // stitch up local phasings
    // (normalized between chunks and save the results)
    for (int i = 1; i < n_jobs; i++) {
        const auto ht_ref_back = get_phased_read_qnames2hp(
                ck_and_vrs[i - 1].ck, ck_and_vrs[i - 1].vr.phasing_breakpoints, N_REF_READS, BACK);
        const auto ht_self_front = get_phased_read_qnames2hp(
                ck_and_vrs[i].ck, ck_and_vrs[i].vr.phasing_breakpoints, N_REF_READS, FRONT);

        int cis = 0;
        int trans = 0;
        for (const auto &[qn, hp_self] : ht_self_front) {
            if (hp_self == HAPTAG_UNPHASED) {
                continue;
            }
            const auto it = ht_ref_back.find(qn);
            if (it != ht_ref_back.cend() && it->second != HAPTAG_UNPHASED) {
                if (it->second == hp_self) {
                    cis++;
                } else {
                    trans++;
                }
            }
        }

        auto &self_qname2hp = ck_and_vrs[i].vr.qname2hp;
        auto &self_ck = ck_and_vrs[i].ck;
        auto &self_vr = ck_and_vrs[i].vr;
        // (if we need to flip:
        //   - update haptags in hashtable (because ck will not be returned)
        //   - update haptags in ck (keep ck and vr in sync; and is needed by genotyping)
        //   - flip genotype strings in ck
        //   - flip genotype strings in the fullinfo variants
        // )
        if (trans > cis) {
            for (const auto &[qn, hp] : self_qname2hp) {
                if (hp != HAPTAG_UNPHASED) {
                    self_qname2hp[qn] = hp ^ 1;
                }
            }

            // (update ck: the read tags and the variants' genotype strings)
            for (uint32_t i_read = 0; i_read < self_ck.reads.size(); i_read++) {
                auto it = self_qname2hp.find(self_ck.qnames[i_read]);
                if (it != self_qname2hp.cend()) {
                    self_ck.reads[i_read].hp = it->second;
                }
            }
            for (auto &var : self_ck.varcalls) {
                std::swap(var.genotype[0], var.genotype[2]);
            }

            // (update fullinfo variants)
            for (auto &fullvar : self_vr.variants) {
                std::swap(fullvar.genotype0[0], fullvar.genotype0[2]);
                std::swap(fullvar.genotype1[0], fullvar.genotype1[2]);
            }
        }
    }

    // save stuff
    for (int i = 0; i < n_jobs; i++) {
        ret[i].vr = std::move(ck_and_vrs[i].vr);
    }
    return ret;
}
}  // namespace

region_string_t parse_region_string2(std::string_view s) {
    region_string_t ret = {
            .is_parse_success = false, .is_whole_chrom = false, .chrom = {}, .start = 0, .end = 0};

    const size_t pos_col = s.find(':');
    if (pos_col == s.npos) {
        ret.is_parse_success = true;
        ret.chrom = s;
        ret.is_whole_chrom = true;
        return ret;
    }
    if (pos_col == s.size() - 1) {
        ret.is_parse_success = false;
        return ret;
    }
    ret.chrom = s.substr(0, pos_col);

    const size_t pos_dash = s.find('-', pos_col + 1);
    if (pos_dash == s.npos) {
        if (pos_col != s.size()) {
        } else {
            ret.is_parse_success = true;
        }
        return ret;
    }

    bool ok = parse_region_integer(s, pos_col + 1, pos_dash, &ret.start);
    if (!ok) {
        return ret;
    }
    ok = parse_region_integer(s, pos_dash + 1, s.size(), &ret.end);
    if (!ok) {
        return ret;
    }

    ret.is_parse_success = true;
    return ret;
}

int util_modify_tag_given_tsv(const std::filesystem::path &fn_bam,
                              const std::filesystem::path &fn_tsv,
                              const std::string &itvl,
                              const int n_threads,
                              const int n_bam_threads,
                              const std::filesystem::path &fn_out) {
    // Debug function. Given a tsv with 2 columns (read name and read's haptag),
    // add/override the HP tag in the bam file accordingly.
    int ret = 0;
    assert(!fn_tsv.native().empty());
    assert(!fn_bam.native().empty());

    std::ifstream fp_tsv(fn_tsv);
    if (!fp_tsv) {
        spdlog::error("[kdys::{}] failed to open input: {}", __func__, fn_tsv.string());
        return 1;
    }

    dorado::secondary::BamFile hf{fn_bam, n_bam_threads};
    spdlog::info("[kdys::{}] itvl: {}", __func__, itvl);
    HtsItrPtr bamitr =
            HtsItrPtr(sam_itr_querys(hf.idx(), hf.hdr(), itvl.c_str()), HtsItrDestructor());

    // read read tags
    std::unordered_map<std::string, int> qname2hp;

    std::string line;
    int warning = 0;
    while (std::getline(fp_tsv, line)) {
        if (line.empty()) {
            continue;
        }
        if (line[0] == '#') {
            continue;
        }
        std::vector<std::string> cols = split_deli_line(line, '\t');
        if (cols.size() != 2 || cols[0].empty() || cols[1].empty()) {
            warning++;
            continue;
        }
        std::string &qname = cols[0];
        if (cols[1].empty()) {
            spdlog::warn("[kdys::{}] read {} has no haptag", __func__, qname);
            continue;
        }
        // insert
        if (qname2hp.find(qname) != qname2hp.end()) {
            spdlog::warn("[kdys::{}] dup read name in input? ({}) (doing nothing)", __func__,
                         qname);
        } else {
            qname2hp[qname] = std::stoi(cols[1]);
        }
    }

    if (KDY_VERBOSE) {
        spdlog::info("[kdys::{}] loaded {} read names", __func__,
                     static_cast<int>(qname2hp.size()));
    }

    // open output file
    BGZF *fp_out = bgzf_open(fn_out.string().c_str(), "w");
    if (!fp_out) {
        if (KDY_VERBOSE) {
            spdlog::error("[kdys::{}] failed to open output file: {}", __func__, fn_out.string());
        }
        return 1;
    }
    if (n_threads > 1) {
        bgzf_mt(fp_out, n_threads, 0 /*unused*/);
    }

    // write
    int stat = bam_hdr_write(fp_out, hf.hdr());
    if (stat != 0) {
        if (KDY_VERBOSE) {
            spdlog::error("[kdys::{}] output bam header write failed", __func__);
        }
        bgzf_close(fp_out);
        return 1;
    }
    int haptag;
    int n_lines = 0;
    int n_lines_seen = 0;
    BamPtr aln = BamPtr(bam_init1(), BamDestructor());
    while (sam_itr_next(hf.fp(), bamitr.get(), aln.get()) >= 0) {
        char *refname = hf.hdr()->target_name[aln->core.tid];
        int start_pos = aln->core.pos;
        char *qn = bam_get_qname(aln.get());

        const auto it = qname2hp.find(qn);
        haptag = (it == qname2hp.cend()) ? HAPTAG_UNPHASED : it->second;
        bam_aux_update_int(aln.get(), "HP", haptag + 1);
        stat = bam_write1(fp_out, aln.get());
        if (stat < 0) {
            spdlog::error("[kdys::{}] failed to write bam entry (ref={} pos={} qn={} newhp={})",
                          __func__, refname, start_pos, qn, haptag);
        } else {
            n_lines++;
        }
        n_lines_seen++;
    }
    aln = {};
    if (KDY_VERBOSE) {
        spdlog::info("[kdys::{}] wrote {} bam lines (saw {} lines)", __func__, n_lines,
                     n_lines_seen);
    }

    // index output file
    bgzf_close(fp_out);
    const std::string fn_bai_out = fn_out.string() + ".bai";
    stat = sam_index_build3(fn_out.string().c_str(), fn_bai_out.c_str(), 0, n_threads);
    if (stat != 0) {
        spdlog::error("[kdys::{}] failed to index output (stat={})", __func__, stat);
        ret = 1;
    }

    return ret;
}

int local_haplotagging(const std::filesystem::path &fn_bam,
                       const std::filesystem::path &fn_bed,
                       const int slice_in_bed,
                       const std::filesystem::path &fn_ref,
                       const std::string_view dbg_region_str,
                       const std::string_view dbg_chrom_str,
                       const std::filesystem::path &fn_vcf,
                       const int chunk_l,
                       const int chunk_stride,
                       const pileup_pars_t &pp,
                       const int n_threads,
                       const int n_bam_threads,
                       const int n_chunks_per_batch,
                       const std::filesystem::path &fn_out_tsv) {
    // return : 0 if all ok, 1 if error

    // open output files
    std::ofstream fp_out_tsv(fn_out_tsv);
    if (!fp_out_tsv.is_open()) {
        spdlog::error("[kdys::{}] failed to open output file {}", __func__, fn_out_tsv.string());
        return 1;
    }

    // for reading the header; threads will open the bam files on their own
    dorado::secondary::BamFile hf{fn_bam, n_bam_threads};

    // will load vcf variants if present
    reference_variants_t refvars;
    if (!fn_vcf.native().empty()) {
        refvars = load_frozen_variants_from_vcf_2ad(fn_vcf);
    }

    for (int i_ref = 0; i_ref < hf.hdr()->n_targets; i_ref++) {
        const char *refname = hf.hdr()->target_name[i_ref];
        const uint32_t ref_l = hf.hdr()->target_len[i_ref];
        if (!dbg_chrom_str.empty() && refname != dbg_chrom_str) {
            continue;  // slow; only enabled when debugging string is given
        }

        // init
        variants_t ht_refvars_empty = {};
        worker_2a2p_pl pl = {
                .n_threads = n_threads,
                .n_chunks_per_batch = n_chunks_per_batch,
                .pp = pp,
                .fn_bam = fn_bam,
                .fn_ref = fn_ref,
                .ht_refvars = fn_vcf.native().empty() ? ht_refvars_empty : refvars[refname],
                .range_i = 0,
                .tot_offset = 0,
                .ranges = {},
                .refname = refname,
                .fp_out = fp_out_tsv,
                .n_bam_threads = n_bam_threads};

        if (!dbg_region_str.empty()) {
            if (region_string_is_sane(dbg_region_str, dbg_region_str.size()) == IS_MALFORMAT) {
                spdlog::error("[kdys::{}] --region was malformatted: {}", __func__, dbg_region_str);
                exit(1);
            }

            region_string_t region = parse_region_string2(dbg_region_str);
            if (!region.is_parse_success) {
                spdlog::error("[kdys::{}] failed to parse region string {}", __func__,
                              dbg_region_str);
                exit(1);
            }
            if (region.is_whole_chrom) {
                region.start = 1;
                region.end = ref_l;
                spdlog::warn(
                        "[kdys::{}] Not slicing in the query range which seems to be a whole "
                        "chromosome. Use BED file "
                        "with --slice-in-bed instead if not intended. The --region is an debug "
                        "option.",
                        __func__);
            }
            if (region.chrom != refname) {
                pl.tot_offset += pl.range_i;
                continue;
            }
            pl.ranges.push_back(u32p_t{.s = region.start, .e = region.end});
            if (KDY_VERBOSE) {
                spdlog::info("[kdys::{}] dbg region pushed: {} {} {}", __func__, region.chrom,
                             (int)region.start, (int)region.end);
            }
        } else if (!fn_bed.native().empty()) {
            const std::vector<u32p_t> loaded_ranges =
                    load_intervals_vars_from_file_one_ref(fn_bed, refname, 0, 0, IS_BED);
            if (slice_in_bed) {
                for (const u32p_t &range : loaded_ranges) {
                    const uint32_t start = range.s;
                    const uint32_t end = range.e;
                    uint32_t i = start;
                    while (i < end) {
                        pl.ranges.push_back({.s = i, .e = static_cast<uint32_t>(i + chunk_l)});
                        i += chunk_stride;
                    }
                }
            } else {
                for (const u32p_t &range : loaded_ranges) {
                    pl.ranges.push_back({.s = range.s, .e = range.e});
                }
            }
        } else {
            uint32_t i = 0;
            while (i < ref_l) {
                pl.ranges.push_back(
                        {.s = static_cast<uint32_t>(i), .e = static_cast<uint32_t>(i + chunk_l)});
                i += chunk_stride;
            }
        }

        // phasing
        local_haplotagging_phase_and_write(&pl);
        pl.tot_offset += pl.range_i;
    }

    fp_out_tsv.close();
    return 0;
}

str2int_t kadayashi_global_phasing_simple_modify_vcf(const std::filesystem::path &fn_ref,
                                                     const std::filesystem::path &fn_bam,
                                                     const std::filesystem::path &fn_in_vcf,
                                                     const std::filesystem::path &fn_out_vcf,
                                                     const int n_threads) {
    reference_variants_t refvars = load_frozen_variants_from_vcf_2ad(fn_in_vcf);
    if (refvars.size() == 0) {
        spdlog::error("[kdys::{}] input vcf is empty? ({})", __func__, fn_in_vcf.string());
        exit(1);
    }

    return kadayashi_global_phasing_simple_modify_vcf1(&refvars, fn_ref, fn_bam, fn_in_vcf,
                                                       fn_out_vcf, n_threads);
}

intervals_t region_strings_to_intervals(dorado::secondary::BamFile &hf,
                                        const int window_size,
                                        const std::vector<std::string> &query_regions) {
    intervals_t ret;
    query_regions_t qregs = region_strings_to_ht(query_regions);
    const bool get_all = qregs.empty();
    for (auto i = 0; i < hf.hdr()->n_targets; i++) {
        const char *chrom = hf.hdr()->target_name[i];
        if (get_all) {
            qregs[chrom] = {};  // instruct the block below to use the whole refseq
        }
        const auto it_qreg = qregs.find(chrom);

        if (it_qreg == qregs.cend()) {
            continue;
        }

        if (window_size == 0) {  // special: don't make chunks, just convert data type
            if (it_qreg->second.size() == 0) {
                ret[chrom] = {};
                continue;
            }
            for (auto &_ : it_qreg->second) {
                ret[chrom].push_back({_.start, _.end});
            }
        } else {  // let's make chunks
            if (hf.hdr()->target_len[i] == 0) {
                spdlog::warn("[kdys::{}] ingoreing reference {} because length is 0", __func__,
                             hf.hdr()->target_name[i]);
                continue;
            }
            if (it_qreg->second.size() == 0) {  // use all the whole length of this reference
                const uint32_t limit_end = hf.hdr()->target_len[i];
                for (uint32_t tmpi = 0; tmpi < limit_end; tmpi += window_size) {  // 0-index
                    ret[chrom].push_back({tmpi, tmpi + window_size > limit_end
                                                        ? limit_end
                                                        : tmpi + window_size});
                }
            } else {
                for (auto &_ : it_qreg->second) {
                    const uint32_t limit_end = _.end;
                    for (uint32_t tmpi = _.start; tmpi < limit_end; tmpi += window_size) {
                        ret[chrom].push_back({tmpi, tmpi + window_size > limit_end
                                                            ? limit_end
                                                            : tmpi + window_size});
                    }
                }
            }
        }
    }
    return ret;
}

struct bed_intereval_t {
    int start;  // in 1-index
    int end;
};
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
                                                    const int bed_flanking) {
    // prep input files and inputs
    dorado::secondary::BamFile hf{fn_bam, 1};
    dorado::hts_io::FastxRandomReader fp_fai{fn_ref};
    intervals_t query_intervals_all = region_strings_to_intervals(hf, window_size, query_regions);

    // prep output files
    std::ofstream fp_out_vcf(fn_out_vcf);
    assert(fp_out_vcf.is_open());

    const std::filesystem::path fn_out_unsr = prefix_out_unsr.string() + ".unsr.list";
    std::ofstream fp_out_unsrlist(fn_out_unsr);
    assert(fp_out_unsrlist.is_open());

    const std::filesystem::path fn_out_kdystsv_apprxglobal =
            prefix_out_unsr.string() + ".apprxglobal.tsv";
    std::ofstream fp_out_kdystsv_apprxglobal(fn_out_kdystsv_apprxglobal);
    assert(fp_out_kdystsv_apprxglobal.is_open());

    const std::filesystem::path fn_out_kdystsv_local = prefix_out_unsr.string() + ".local.tsv";
    std::ofstream fp_out_kdystsv_local(fn_out_kdystsv_local);
    assert(fp_out_kdystsv_local.is_open());

    const std::filesystem::path fn_out_bed = prefix_out_unsr.string() + ".bed";
    std::ofstream fp_out_bed(fn_out_bed);
    assert(fp_out_bed.is_open());

    // (write vcf header)
    varcall_write_simple_vcf_header(fp_out_vcf, hf);

    // prep holders
    str2int_t qname2hp_all;  // approx. global, for return

    // prep threading
    const int n_threads_inner = n_threads > 4 ? 4 : 1;
    const int n_threads_job = n_threads > 4 ? n_threads / n_threads_inner : n_threads;

    uint32_t binchunkID = 0;  // kadayashi tsv
    for (int i_ref = 0; i_ref < hf.hdr()->n_targets; i_ref++) {
        const char *chrom = hf.hdr()->target_name[i_ref];
        const int chrom_size = hf.hdr()->target_len[i_ref];
        if (!query_intervals_all.empty() &&
            query_intervals_all.find(chrom) == query_intervals_all.cend()) {
            continue;
        }

        std::vector<std::pair<uint32_t, uint32_t>> &query_intervals = query_intervals_all[chrom];
        if (query_intervals.empty()) {                   // use whole chromosome
            query_intervals.push_back({0, chrom_size});  // 0-index
        }

        // call variants
        std::vector<varcall_result_and_localphasinght_t> varcall_results =
                kadayashi_phase_and_varcall_multiregionthreaded(
                        fn_ref, fn_bam, n_threads_job, n_threads_inner, chrom, query_intervals,
                        readgroup, disable_interval_expansion, min_base_quality,
                        min_varcall_coverage, min_varcall_fraction, max_clipping, min_strand_cov,
                        min_strand_cov_frac, max_gapcompressed_seqdiv, ambig_ref);

        // helper: convert phasing breakpoints into a sorted array
        // which will be queried for phaseblock IDs.
        std::vector<uint32_t> breakpoints_arr;
        for (uint32_t i_itvl = 0; i_itvl < varcall_results.size(); i_itvl++) {
            for (auto &[pos, _] : varcall_results[i_itvl].vr.phasing_breakpoints) {
                breakpoints_arr.push_back(pos);  // 0-index
            }
        }
        std::sort(breakpoints_arr.begin(), breakpoints_arr.end());
        for (auto &pos : breakpoints_arr) {
            spdlog::info("[kdys::{}] phasing breakpoint at {}", __func__, (int)pos);
        }

        // write vcf and unsure positions
        std::vector<uint32_t> unsure_poss;
        std::vector<bed_intereval_t> unsure_intervals;
        uint32_t phaseblockID_fallback = 0;
        for (uint32_t i = 0; i < query_intervals.size(); i++) {
            const auto &itvl = query_intervals[i];
            for (const auto &fullvar : varcall_results[i].vr.variants) {
                if (!fullvar.is_valid) {
                    continue;
                }

                // vcf
                const std::string vcf_line = make_vcf_line_given_variant_fullinfo_t(
                        chrom, itvl.first, itvl.second, breakpoints_arr, fullvar,
                        phaseblockID_fallback);
                fp_out_vcf << fmt::format("{:s}", vcf_line);

                // unsure list
                if (!fullvar.is_confident) {
                    const std::vector<size_t> allele_lens = {
                            fullvar.ref_allele_seq0.size(),
                            fullvar.alt_allele_seq0.size(),
                            fullvar.ref_allele_seq1.size(),
                            fullvar.alt_allele_seq1.size(),
                    };
                    const int longest_allele_len = static_cast<int>(
                            *std::max_element(allele_lens.begin(), allele_lens.end()));
                    const int pos = static_cast<int>(fullvar.pos0);
                    fp_out_unsrlist
                            << fmt::format("{:s}\t{:d}\t{:d}\n", chrom, pos, longest_allele_len);

                    const int left = std::max(1, pos - bed_flanking);
                    const int right = std::min(pos + bed_flanking, chrom_size + 1);
                    if (unsure_intervals.empty()) {
                        unsure_intervals.push_back({.start = left, .end = right});
                    } else {
                        if (left <= unsure_intervals.back().end) {
                            unsure_intervals.back().end = right;
                        } else {
                            unsure_intervals.push_back({.start = left, .end = right});
                        }
                    }
                }
            }

            // write kadayashi tsv
            const uint32_t chunk_start = query_intervals[i].first;
            const uint32_t chunk_end = query_intervals[i].second;
            local_haptagging_write_tsv2(fp_out_kdystsv_local, hf, (int)binchunkID, chrom,
                                        chunk_start, chunk_end, varcall_results[i].qname2hp_local);
            local_haptagging_write_tsv2(fp_out_kdystsv_apprxglobal, hf, (int)binchunkID, chrom,
                                        chunk_start, chunk_end, varcall_results[i].vr.qname2hp);
            assert(binchunkID < INTMAX_MAX - 1);
            binchunkID++;

            // collect global phasing (for them to be returned)
            for (const auto &[qn, hp] : varcall_results[i].vr.qname2hp) {
                qname2hp_all[qn] = hp;
            }
        }  // iter through query intervals on one chrom

        // write bed file with fixed padding on each side
        for (const auto &_ : unsure_intervals) {
            fp_out_bed << fmt::format("{:s}\t{:d}\t{:d}\n", chrom, _.start, _.end);
        }
    }  // iter through chroms
    fp_out_vcf.close();
    fp_out_unsrlist.close();
    fp_out_kdystsv_apprxglobal.close();
    fp_out_kdystsv_local.close();
    fp_out_bed.close();

    // write kadayashi bin file
    std::filesystem::path fn_out_kdysbin_apprxglobal = fn_out_kdystsv_apprxglobal;
    std::filesystem::path fn_out_kdysbin_local = fn_out_kdystsv_local;
    fn_out_kdysbin_apprxglobal += ".bin";
    fn_out_kdysbin_local += ".bin";
    write_binary_given_tsv(fn_out_kdystsv_apprxglobal, fn_out_kdysbin_apprxglobal);
    write_binary_given_tsv(fn_out_kdystsv_local, fn_out_kdysbin_local);

    return qname2hp_all;
}

// for local realn testing with spoa
std::vector<std::string> bam_region_to_seqs(const std::filesystem::path &fn_ref,
                                            const std::filesystem::path &fn_bam,
                                            const std::string &interval_string,
                                            const int n_threads,
                                            const std::filesystem::path &fn_out_fa) {
    // The first seq in return is the region's reference sequence,
    // followed by reads in the sorted bam's order.
    // Read names are not returned.
    std::vector<std::string> ret;
    std::ofstream fp_out_fa;
    if (!fn_out_fa.native().empty()) {
        fp_out_fa.open(fn_out_fa);
        assert(fp_out_fa.is_open());
    }

    dorado::hts_io::FastxRandomReader fp_fai{fn_ref};
    const std::string refseq_s = fp_fai.fetch_seq(interval_string.c_str());

    ret.push_back(refseq_s);
    if (fp_out_fa) {
        fp_out_fa << fmt::format(">ref {}\n{}\n", interval_string, ret.back());
    }

    dorado::secondary::BamFile hf{fn_bam, n_threads};
    HtsItrPtr bamitr = HtsItrPtr(sam_itr_querys(hf.idx(), hf.hdr(), interval_string.c_str()),
                                 HtsItrDestructor());
    BamPtr aln = BamPtr(bam_init1(), BamDestructor());  // use local buffer instead of hf->aln

    uint32_t ref_start = 0;
    uint32_t ref_end = 0;
    int region_size = 0;
    const region_string_t region = parse_region_string2(interval_string);
    if (!region.is_parse_success) {
        spdlog::error("[kdys::{}] failed to parse region string {}", __func__, interval_string);
        return {};
    } else {
        region_size = region.end - region.start;
        ref_start = region.start;
        ref_end = region.end;
    }

    int readID = 0;
    std::string seq;
    while (sam_itr_next(hf.fp(), bamitr.get(), aln.get()) >= 0) {
        int flag = aln.get()->core.flag;
        int seq_len = aln.get()->core.l_qseq;
        uint8_t *seqdata = bam_get_seq(aln.get());
        uint32_t *cigar = bam_get_cigar(aln.get());

        seq.resize(seq_len, 0);
        for (int i = 0; i < seq_len; i++) {
            char base = seq_nt16_str[bam_seqi(seqdata, i)];
            seq[i] = base;  // note: strand is already accounted for
        }

        read_t read;
        read.ID = readID;
        read.start_pos = static_cast<uint32_t>(aln.get()->core.pos);
        read.end_pos = static_cast<uint32_t>(bam_endpos(aln.get()));
        read.strand = !!(flag & 16);
        const bool parse_ok =
                parse_variants_for_one_read(aln.get(), read.vars, 0, &read.left_clip_len,
                                            &read.right_clip_len, 0 /*SNPonly?*/, NULL);
        if (!parse_ok) {
            continue;
        }
        std::stable_sort(read.vars.begin(), read.vars.end());

        int *offset;
        int offset1 = 0;  // bases to skip/include up to the read's alignment start position
        int offset2 =
                region_size;  // bases to skip/include between alignment start and end positions
        uint32_t op = bam_cigar_op(cigar[0]);
        uint32_t op_l = bam_cigar_oplen(cigar[0]);
        if (op == BAM_CSOFT_CLIP) {
            offset1 += op_l;
        }
        for (const qa_t &vars : read.vars) {
            uint32_t var_pos = vars.pos;
            if (var_pos < ref_end) {
                if (var_pos < ref_start) {
                    offset = &offset1;
                } else {
                    offset = &offset2;
                }
                uint32_t var_size = vars.allele.size() - 1;
                uint8_t op = vars.allele.back();
                if (op == VAR_OP_I) {
                    *offset += var_size;
                } else if (op == VAR_OP_D) {
                    *offset -= var_size;
                }
            } else {
                break;
            }
        }

        if (ref_start > read.start_pos) {
            offset1 += (ref_start - read.start_pos);
        }

        int seq_substart = offset1 > 0 ? offset1 - 1 : 0;
        int seq_subend = std::min(offset1 + offset2, seq_len);
        if (seq_substart >= seq_subend) {
            ret.push_back("");
        } else {
            ret.push_back(seq.substr(seq_substart, seq_subend - seq_substart));
        }

        if (fp_out_fa) {
            fp_out_fa << fmt::format(">{:s} {:d}-{:d}\n{:s}\n", bam_get_qname(aln.get()), offset1,
                                     offset1 + offset2, ret.back());
        }

        readID++;
    }

    if (fp_out_fa.is_open()) {
        fp_out_fa.close();
    }

    return ret;
}

}  // namespace kadayashi
