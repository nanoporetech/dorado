#include "TestUtils.h"
#include "hts_utils/FastxRandomReader.h"
#include "hts_utils/hts_file.h"
#include "hts_utils/hts_types.h"
#include "secondary/common/bam_file.h"
#include "secondary/features/medaka_read_matrix.h"
#include "secondary/kadayashi/blocked_bloom_filter.h"
#include "secondary/kadayashi/local_haplotagging.h"
#include "secondary/kadayashi/sequence_utility.h"
#include "secondary/kadayashi/types.h"
#include "utils/cigar.h"

#include <catch2/catch_test_macros.hpp>
#include <htslib/faidx.h>
#include <htslib/khash.h>
#include <htslib/khash_str2int.h>
#include <htslib/sam.h>
#include <stdint.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define TEST_GROUP "[KadayashiInterfaceTest]"

namespace kadayashi::tests {

namespace {
bool compare_haptags(const std::unordered_map<std::string, int32_t> &result,
                     const std::unordered_map<std::string, int32_t> &expected) {
    const bool is_good = (expected == result);

    if (!is_good) {
        // Try swapping the phases because they are arbitrary: 0->1 and 1->0.
        std::unordered_map<std::string, int32_t> relabeled_result;
        for (const auto &[key, hap] : result) {
            const int32_t new_hap = (hap == 1) ? 0 : 1;
            relabeled_result[key] = new_hap;
        }
        return relabeled_result == expected;
    }

    return is_good;
}

struct SyntheticBamRecord {
    std::string_view qname{};
    int32_t tid{0};
    int32_t pos{0};
    uint16_t flag{0};
    uint8_t mapq{60};
    std::string_view cigar{};
    std::string_view seq{};
    std::string_view qual{};
    std::vector<int8_t> dwell_tag{};
    std::string_view md{};
    std::optional<int32_t> nm{0};
    std::optional<int32_t> hp_tag{};
    std::string readgroup{"rg0"};
};

const std::vector<std::string> EMPTY_DTYPES;
const std::string EMPTY_STRING;

std::vector<uint32_t> parse_cigar_from_string_hts(const std::string_view cigar) {
    const std::vector<dorado::CigarOp> parsed = dorado::parse_cigar_from_string(cigar);
    std::vector<uint32_t> ret(std::size(parsed));
    for (size_t i = 0; i < std::size(parsed); ++i) {
        const dorado::CigarOp cig = parsed[i];
        ret[i] = bam_cigar_gen(cig.len, static_cast<int8_t>(cig.op));
    }
    return ret;
}

dorado::SamHdrPtr make_synthetic_bam_header(
        const std::span<const std::pair<std::string, std::string>> targets,
        const std::span<const SyntheticBamRecord> records) {
    std::string text = "@HD\tVN:1.6\tSO:unknown\n";
    for (const auto &[name, seq] : targets) {
        text += "@SQ\tSN:" + name + "\tLN:" + std::to_string(std::size(seq)) + "\n";
    }

    std::unordered_set<std::string> readgroups;
    for (const SyntheticBamRecord &record : records) {
        readgroups.insert(record.readgroup);
    }
    for (const std::string &rg : readgroups) {
        text += "@RG\tID:" + rg + "\n";
    }

    return dorado::SamHdrPtr{sam_hdr_parse(std::size(text), text.c_str())};
}

dorado::BamPtr make_synthetic_bam_record(const SyntheticBamRecord &record) {
    const std::vector<uint32_t> cigar_vec = parse_cigar_from_string_hts(record.cigar);
    dorado::BamPtr ret{bam_init1()};

    bam_set1(ret.get(), std::size(record.qname), std::data(record.qname), record.flag, record.tid,
             record.pos, record.mapq, std::size(cigar_vec), std::data(cigar_vec), -1, -1, 0,
             std::size(record.seq), std::data(record.seq),
             std::empty(record.qual) ? nullptr : std::data(record.qual), 0);

    // Move table specification:
    // https://software-docs.nanoporetech.com/dorado/latest/basecaller/move_table
    // Tag format: `mv:B:c,[block_stride],[signal_block_move_list]`.
    if (!std::empty(record.dwell_tag)) {
        std::vector<uint8_t> aux_data;
        aux_data.emplace_back('c');

        const uint32_t n = static_cast<uint32_t>(std::size(record.dwell_tag));
        const uint8_t *n_bytes = reinterpret_cast<const uint8_t *>(&n);
        aux_data.insert(std::end(aux_data), n_bytes, n_bytes + sizeof(n));

        for (const int8_t val : record.dwell_tag) {
            aux_data.emplace_back(static_cast<uint8_t>(val));
        }

        const int32_t aux_data_len = static_cast<int32_t>(std::size(aux_data));
        bam_aux_append(ret.get(), "mv", 'B', aux_data_len, std::data(aux_data));
    }

    if (!std::empty(record.md)) {
        const std::string md{record.md};
        const int32_t md_len = static_cast<int32_t>(std::size(md) + 1);
        bam_aux_append(ret.get(), "MD", 'Z', md_len, reinterpret_cast<const uint8_t *>(md.c_str()));
    }

    if (record.nm) {
        bam_aux_append(ret.get(), "NM", 'i', sizeof(int32_t),
                       reinterpret_cast<const uint8_t *>(&(*record.nm)));
    }

    if (record.hp_tag) {
        bam_aux_append(ret.get(), "HP", 'i', sizeof(int32_t),
                       reinterpret_cast<const uint8_t *>(&(*record.hp_tag)));
    }

    if (!record.readgroup.empty()) {
        bam_aux_append(
                ret.get(), "RG", 'Z', record.readgroup.size() + 1,
                reinterpret_cast<const uint8_t *>(const_cast<char *>(record.readgroup.c_str())));
    }

    return ret;
}

void write_synthetic_bam(const std::filesystem::path &out_fn,
                         const std::span<const std::pair<std::string, std::string>> targets,
                         const std::span<const SyntheticBamRecord> records) {
    dorado::utils::HtsFile hts_file(out_fn.string(), dorado::utils::HtsFile::OutputMode::BAM, 1,
                                    true);
    dorado::SamHdrPtr header = make_synthetic_bam_header(targets, records);
    hts_file.set_header(header.get());
    for (const SyntheticBamRecord &record : records) {
        dorado::BamPtr bam_record = make_synthetic_bam_record(record);
        hts_file.write(bam_record.get());
    }
    hts_file.finalise([](size_t) {});
}

kadayashi::MedakaFeatureMatrixOptions make_medaka_feature_matrix_options(
        const bool include_dwells,
        const double min_snp_accuracy,
        const int32_t max_reads,
        const bool disable_read_packing) {
    return {.include_dwells = include_dwells,
            .include_haplotype_column = false,
            .include_snp_qv = false,
            .min_mapq = 1,
            .num_dtypes = 1,
            .dtypes = EMPTY_DTYPES,
            .tag_name = EMPTY_STRING,
            .tag_value = 0,
            .tag_keep_missing = false,
            .readgroup = "",
            .disable_read_packing = disable_read_packing,
            .hap_source = kadayashi::FORCE_UNPHASED,
            .max_reads = max_reads,
            .right_align_insertions = true,
            .min_snp_accuracy = min_snp_accuracy};
}

std::vector<int8_t> make_simple_move_table(const int32_t sequence_length) {
    std::vector<int8_t> ret(static_cast<size_t>(sequence_length) + 1, 1);
    ret.front() = 1;
    return ret;
}

dorado::secondary::ReadAlignmentData run_feature_matrix_test(
        const std::span<const SyntheticBamRecord> records,
        const std::string &refseq,
        const uint32_t ref_start,
        const uint32_t ref_end,
        const kadayashi::MedakaFeatureMatrixOptions &options) {
    const TempDir temp_dir = make_temp_dir("kadayashi_featmatgen");
    const std::filesystem::path temp_in_bam_fn = temp_dir.m_path / "in.aln.bam";
    const std::vector<std::pair<std::string, std::string>> targets{{"ref", refseq}};
    write_synthetic_bam(temp_in_bam_fn, targets, records);

    dorado::secondary::BamFile bam_file(temp_in_bam_fn, 1);
    return kadayashi::gen_medaka_feature_matrix_wrapper(bam_file, "ref", ref_start, ref_end, {},
                                                        options);
}

int8_t get_feature_matrix_value(const dorado::secondary::ReadAlignmentData &result,
                                const int32_t pos,
                                const int32_t lane,
                                const int32_t feature) {
    return result.matrix[(pos * result.buffer_reads + lane) * result.featlen + feature];
}

void compare_position_vector(const std::vector<int64_t> &result,
                             const std::vector<int64_t> &expected,
                             const std::string_view name) {
    CATCH_REQUIRE(result.size() == expected.size());

    for (size_t i = 0; i < result.size(); ++i) {
        if (result[i] != expected[i]) {
            CATCH_CAPTURE(name, i, result[i], expected[i]);
            CATCH_CHECK(result[i] == expected[i]);
        }
    }
}

bool has_non_zero_feature_value(const dorado::secondary::ReadAlignmentData &result,
                                const int32_t feature) {
    for (int32_t pos = 0; pos < result.n_pos; ++pos) {
        for (int32_t lane = 0; lane < result.n_reads; ++lane) {
            if (get_feature_matrix_value(result, pos, lane, feature) != 0) {
                return true;
            }
        }
    }

    return false;
}

void compare_feature_matrix_logical_data(const dorado::secondary::ReadAlignmentData &result,
                                         const dorado::secondary::ReadAlignmentData &expected) {
    CATCH_REQUIRE(result.n_pos == expected.n_pos);
    CATCH_REQUIRE(result.n_reads == expected.n_reads);
    CATCH_REQUIRE(result.featlen == expected.featlen);
    CATCH_REQUIRE(result.num_dtypes == expected.num_dtypes);
    CATCH_REQUIRE(result.buffer_reads >= result.n_reads);
    CATCH_REQUIRE(expected.buffer_reads >= expected.n_reads);

    compare_position_vector(result.major, expected.major, "major");
    compare_position_vector(result.minor, expected.minor, "minor");

    CATCH_CHECK(result.read_ids_left == expected.read_ids_left);
    CATCH_CHECK(result.read_ids_right == expected.read_ids_right);

    // Stop comparing after this many bad results.
    constexpr size_t MAX_REPORTED_MISMATCHES = 10;

    size_t n_mismatches = 0;
    for (int32_t pos = 0; pos < result.n_pos; ++pos) {
        for (int32_t lane = 0; lane < result.n_reads; ++lane) {
            for (int32_t feature = 0; feature < result.featlen; ++feature) {
                const int32_t result_value =
                        static_cast<int32_t>(get_feature_matrix_value(result, pos, lane, feature));
                const int32_t expected_value = static_cast<int32_t>(
                        get_feature_matrix_value(expected, pos, lane, feature));
                if (result_value != expected_value) {
                    CATCH_CAPTURE(pos, lane, feature, result_value, expected_value);
                    CATCH_CHECK(result_value == expected_value);
                    ++n_mismatches;
                    if (n_mismatches >= MAX_REPORTED_MISMATCHES) {
                        return;
                    }
                }
            }
        }
    }

    CATCH_CHECK(n_mismatches == 0);
}

}  // namespace

CATCH_TEST_CASE("kadayashi blocked bloom filter basic operation", TEST_GROUP) {
    kadayashi::BlockedBloomFilter bf(4, 16);  // tiny
    const bool enable1 = bf.enable();
    const bool enable2 = bf.enable();
    CATCH_CHECK(enable1);
    CATCH_CHECK_FALSE(enable2);

    bf.insert(42);
    bool real_is_found = bf.query(42);
    bool fake_is_found = bf.query(43);
    CATCH_CHECK(real_is_found);
    CATCH_CHECK_FALSE(fake_is_found);
}

CATCH_TEST_CASE("kadayashi max of u32 arr", TEST_GROUP) {
    // Tie breaking actually doesn't matter, though it has been picking the first winner.
    const std::array<uint32_t, 5> d = {3, 2, 6, 6, 1};
    int idx = 9;
    uint32_t m;
    const bool cmp_ok = kadayashi::max_of_u32_arr(d, &idx, &m);
    CATCH_CHECK(cmp_ok);
    CATCH_CHECK(idx == 2);
    CATCH_CHECK(m == 6);

    // Error case: empty
    bool cmp_ok2 = kadayashi::max_of_u32_arr({}, nullptr, nullptr);
    CATCH_CHECK_FALSE(cmp_ok2);

    // Error case: length 1
    const std::array<uint32_t, 1> d2 = {3};
    const bool cmp_ok3 = kadayashi::max_of_u32_arr(d2, &idx, &m);
    CATCH_CHECK(cmp_ok3);
    CATCH_CHECK(idx == 0);
    CATCH_CHECK(m == 3);
}

CATCH_TEST_CASE("kadayashi variant pileup downsampling", TEST_GROUP) {
    const TempDir temp_dir = make_temp_dir("kadayashi_short_read_downsample");
    const std::filesystem::path temp_in_bam_fn = temp_dir.m_path / "in.aln.bam";

    const std::string refseq(100, 'A');
    const std::vector<std::pair<std::string, std::string>> targets{{"ref", refseq}};

    const std::string read_seq = [] {
        std::string seq(50, 'A');
        seq[10] = 'T';
        return seq;
    }();

    constexpr int32_t NUM_READS_TARGET = kadayashi::DOWNSAMPLE_READCAP;  // target read depth
    constexpr int32_t NUM_READS_REDUNDANT = 50;
    constexpr int32_t NUM_READS = NUM_READS_TARGET + NUM_READS_REDUNDANT;  // add some extra depth
    std::vector<std::string> qnames;
    std::vector<SyntheticBamRecord> records;
    qnames.reserve(NUM_READS);
    records.reserve(NUM_READS);

    for (int32_t i = 0; i < NUM_READS; ++i) {
        qnames.emplace_back("read_" + std::to_string(i));
        records.push_back({
                .qname = qnames.back(),
                .pos = 0,
                .flag = static_cast<uint16_t>((i % 2) == 0 ? 0 : BAM_FREVERSE),
                .mapq = 60,
                .cigar = "50M",
                .seq = read_seq,
                .md = "10A39",
                .nm = 1,
        });
    }

    write_synthetic_bam(temp_in_bam_fn, targets, records);
    dorado::secondary::BamFile bam_file(temp_in_bam_fn, 1);
    dorado::secondary::BamFileView bam_view = bam_file.get_view();

    kadayashi::pileup_pars_t pp{};
    pp.min_base_quality = 0;
    pp.min_mapq = 1;
    pp.min_varcall_coverage = 1;
    pp.min_varcall_fraction = 0.0f;
    pp.max_clipping = 100000;
    pp.min_strand_cov = 1;
    pp.min_strand_cov_frac = 0.0f;
    pp.retain_het_only = false;
    pp.disable_low_complexity_masking = true;
    pp.disable_region_expansion = true;

    const kadayashi::chunk_t result =
            kadayashi::variant_pileup_ht(bam_view, {}, nullptr, nullptr, "ref", 0, 100, pp);

    CATCH_CHECK(result.is_valid);
    CATCH_CHECK(result.qnames.size() <= NUM_READS_TARGET);
    CATCH_CHECK(result.reads.size() <= NUM_READS_TARGET);
}

CATCH_TEST_CASE("kadayashi phasing and varcall pileup readgroup filtering", TEST_GROUP) {
    const TempDir temp_dir = make_temp_dir("kadayashi_readgroup_filter");
    const std::filesystem::path temp_in_bam_fn = temp_dir.m_path / "in.aln.bam";

    const std::string refseq(100, 'A');
    const std::vector<std::pair<std::string, std::string>> targets{{"ref", refseq}};

    const std::string read_seq = [] {
        std::string seq(50, 'A');
        seq[10] = 'T';
        return seq;
    }();

    constexpr int32_t NUM_READS = 10;
    std::vector<std::string> qnames;
    std::vector<SyntheticBamRecord> records;
    qnames.reserve(NUM_READS * 2);   // we will generate two read groups
    records.reserve(NUM_READS * 2);  // (same as above)

    for (int rg = 0; rg < 2; rg++) {
        for (int32_t i = 0; i < NUM_READS; ++i) {
            qnames.emplace_back("read_" + std::to_string(i + NUM_READS * rg));
            records.push_back({.qname = qnames.back(),
                               .pos = 0,
                               .flag = static_cast<uint16_t>((i % 2) == 0 ? 0 : BAM_FREVERSE),
                               .mapq = 60,
                               .cigar = "50M",
                               .seq = read_seq,
                               .md = "10A39",
                               .nm = 1,
                               .readgroup = "rg" + std::to_string(rg)});
        }
    }

    write_synthetic_bam(temp_in_bam_fn, targets, records);
    dorado::secondary::BamFile bam_file(temp_in_bam_fn, 1);
    dorado::secondary::BamFileView bam_view = bam_file.get_view();

    kadayashi::pileup_pars_t pp{};
    pp.min_base_quality = 0;
    pp.min_mapq = 1;
    pp.min_varcall_coverage = 1;
    pp.min_varcall_fraction = 0.0f;
    pp.max_clipping = 100000;
    pp.min_strand_cov = 1;
    pp.min_strand_cov_frac = 0.0f;
    pp.retain_het_only = false;
    pp.disable_low_complexity_masking = true;
    pp.disable_region_expansion = true;
    pp.readgroup = "rg0";

    const kadayashi::chunk_t result =
            kadayashi::variant_pileup_ht(bam_view, {}, nullptr, nullptr, "ref", 0, 100, pp);
    CATCH_CHECK(result.is_valid);
    CATCH_CHECK(result.qnames.size() == NUM_READS);  // should select only one readgroup
    CATCH_CHECK(result.reads.size() == NUM_READS);

    pp.readgroup = "";
    const kadayashi::chunk_t result_both =
            kadayashi::variant_pileup_ht(bam_view, {}, nullptr, nullptr, "ref", 0, 100, pp);
    CATCH_CHECK(result_both.is_valid);
    CATCH_CHECK(result_both.qnames.size() == NUM_READS * 2);  // should select both readgroups
    CATCH_CHECK(result_both.reads.size() == NUM_READS * 2);
}

CATCH_TEST_CASE("kadayashi phasing, normal case", TEST_GROUP) {
    // Input data.
    const std::filesystem::path test_data_dir = get_data_dir("variant") / "test-02-supertiny";
    const std::filesystem::path fn_bam = test_data_dir / "in.aln.bam";
    const std::filesystem::path fn_ref = test_data_dir / "in.ref.fasta.gz";

    const std::unordered_map<std::string, int32_t> expected{
            {"1e70cda3-c41f-4d19-9c14-94d8d64e619c", 1},
            {"61ab09d6-072f-4ab2-b14b-b0a1e38a3419", 1},
            {"563ecca1-30dd-4dd9-991a-d417d827c803", 0},
            {"4fd81aa2-cb77-4994-a8a5-70e6228f255e", 0},
            {"a27cad27-2297-40d4-8666-40a4742eb2ed", 1},
            {"e0af6c87-8655-4603-97b7-0ad5ba860df2", 0},
            {"7d23577c-5c93-4d41-83bd-b652e687deee", 0},
            {"627ea9e1-5204-4a2c-ae54-1e1be8bbbbe6", 1},
            {"ac863a7d-932e-42fa-91c1-7814d7f810f9", 1},
            {"b4139858-e420-4780-94e6-375542c2d2e8", 0},
            {"dbe9785a-fa25-454c-9960-fd65fb99a040", 1},
            {"3fdc1b9b-7186-411e-af92-e93a1086754c", 1},
            {"7b2095d4-08f7-448d-aa9d-55c9568fb49d", 1},
            {"c488f4c5-1639-4be1-92f6-948f29b7d822", 1},
            {"02551418-20c9-4b4b-9d1b-9bee36342895", 1},
            {"de45db56-e704-4524-af88-06a2f98c270e", 1},
            {"49b05d0d-97ac-449e-804b-35b35e05ce28", 0},
            {"e7e27cb5-1144-49dd-8ec4-09a75937a091", 0},
            {"3d7a9813-67be-4b84-b66a-0269aa108340", 1},
            {"d5560893-59c8-417c-a929-d62b4d19a1ca", 1},
    };

    // Open the input files.
    dorado::secondary::BamFile bam_reader(fn_bam, 1);
    dorado::hts_io::FastxRandomReader fastx_reader(fn_ref);

    CATCH_REQUIRE(bam_reader.fp());
    CATCH_REQUIRE(bam_reader.idx());
    CATCH_REQUIRE(bam_reader.hdr());
    CATCH_REQUIRE(fastx_reader.get_raw_faidx_ptr());

    constexpr bool DISABLE_INTERVAL_EXPANSION = false;
    constexpr int32_t MIN_BASE_QUALITY = 5;
    constexpr int32_t MIN_VARCALL_COVERAGE = 5;
    constexpr float MIN_VARCALL_FRACTION = 0.2f;
    constexpr int32_t MAX_CLIPPING = 100000;
    constexpr int32_t MIN_STRAND_COV = 3;
    constexpr float MIN_STRAND_COV_FRAC = 0.03f;
    constexpr float MAX_GAPCOMPRESSED_SEQDIV = 0.1f;

    CATCH_SECTION("kadayashi_simple_single_region_wrapper") {
        const std::unordered_map<std::string, int32_t> result =
                kadayashi::kadayashi_simple_single_region_wrapper(
                        bam_reader.fp(), bam_reader.idx(), bam_reader.hdr(),
                        fastx_reader.get_raw_faidx_ptr(), "chr20", 0, 9999, "" /*readgroup*/,
                        DISABLE_INTERVAL_EXPANSION, MIN_BASE_QUALITY, MIN_VARCALL_COVERAGE,
                        MIN_VARCALL_FRACTION, MAX_CLIPPING, MIN_STRAND_COV, MIN_STRAND_COV_FRAC,
                        MAX_GAPCOMPRESSED_SEQDIV);
        CATCH_CHECK(compare_haptags(result, expected));
    }
}

CATCH_TEST_CASE("kadayashi phasing, empty region", TEST_GROUP) {
    // Input data.
    const std::filesystem::path test_data_dir = get_data_dir("variant") / "test-02-supertiny";
    const std::filesystem::path fn_bam = test_data_dir / "in.aln.bam";
    const std::filesystem::path fn_ref = test_data_dir / "in.ref.fasta.gz";

    const std::unordered_map<std::string, int32_t> expected{};

    // Open the input files.
    dorado::secondary::BamFile bam_reader(fn_bam, 1);
    dorado::hts_io::FastxRandomReader fastx_reader(fn_ref);

    CATCH_REQUIRE(bam_reader.fp());
    CATCH_REQUIRE(bam_reader.idx());
    CATCH_REQUIRE(bam_reader.hdr());
    CATCH_REQUIRE(fastx_reader.get_raw_faidx_ptr());

    constexpr bool DISABLE_INTERVAL_EXPANSION = false;
    constexpr int32_t MIN_BASE_QUALITY = 5;
    constexpr int32_t MIN_VARCALL_COVERAGE = 5;
    constexpr float MIN_VARCALL_FRACTION = 0.2f;
    constexpr int32_t MAX_CLIPPING = 100000;
    constexpr int32_t MIN_STRAND_COV = 3;
    constexpr float MIN_STRAND_COV_FRAC = 0.03f;
    constexpr float MAX_GAPCOMPRESSED_SEQDIV = 0.1f;

    CATCH_SECTION("kadayashi_simple_single_region_wrapper empty region") {
        // UUT.
        const std::unordered_map<std::string, int32_t> result =
                kadayashi::kadayashi_simple_single_region_wrapper(
                        bam_reader.fp(), bam_reader.idx(), bam_reader.hdr(),
                        fastx_reader.get_raw_faidx_ptr(), "chr20", 200000, 200001, "" /*readgroup*/,
                        DISABLE_INTERVAL_EXPANSION, MIN_BASE_QUALITY, MIN_VARCALL_COVERAGE,
                        MIN_VARCALL_FRACTION, MAX_CLIPPING, MIN_STRAND_COV, MIN_STRAND_COV_FRAC,
                        MAX_GAPCOMPRESSED_SEQDIV);
        CATCH_CHECK(compare_haptags(result, expected));
    }

    CATCH_SECTION("kadayashi_simple_single_region_wrapper bad coordinate span") {
        // UUT.
        const std::unordered_map<std::string, int32_t> result =
                kadayashi::kadayashi_simple_single_region_wrapper(
                        bam_reader.fp(), bam_reader.idx(), bam_reader.hdr(),
                        fastx_reader.get_raw_faidx_ptr(), "chr20", 200000, 199999, "" /*readgroup*/,
                        DISABLE_INTERVAL_EXPANSION, MIN_BASE_QUALITY, MIN_VARCALL_COVERAGE,
                        MIN_VARCALL_FRACTION, MAX_CLIPPING, MIN_STRAND_COV, MIN_STRAND_COV_FRAC,
                        MAX_GAPCOMPRESSED_SEQDIV);
        CATCH_CHECK(compare_haptags(result, expected));
    }
}

CATCH_TEST_CASE("kadayashi phasing nonexistent chromosome", TEST_GROUP) {
    // Input data.
    const std::filesystem::path test_data_dir = get_data_dir("variant") / "test-02-supertiny";
    const std::filesystem::path fn_bam = test_data_dir / "in.aln.bam";
    const std::filesystem::path fn_ref = test_data_dir / "in.ref.fasta.gz";

    const std::unordered_map<std::string, int32_t> expected{};

    // Open the input files.
    dorado::secondary::BamFile bam_reader(fn_bam, 1);
    dorado::hts_io::FastxRandomReader fastx_reader(fn_ref);

    CATCH_REQUIRE(bam_reader.fp());
    CATCH_REQUIRE(bam_reader.idx());
    CATCH_REQUIRE(bam_reader.hdr());
    CATCH_REQUIRE(fastx_reader.get_raw_faidx_ptr());

    constexpr bool DISABLE_INTERVAL_EXPANSION = false;
    constexpr int32_t MIN_BASE_QUALITY = 5;
    constexpr int32_t MIN_VARCALL_COVERAGE = 5;
    constexpr float MIN_VARCALL_FRACTION = 0.2f;
    constexpr int32_t MAX_CLIPPING = 100000;
    constexpr int32_t MIN_STRAND_COV = 3;
    constexpr float MIN_STRAND_COV_FRAC = 0.03f;
    constexpr float MAX_GAPCOMPRESSED_SEQDIV = 0.1f;

    CATCH_SECTION("kadayashi_simple_single_region_wrapper empty region") {
        const std::unordered_map<std::string, int32_t> result =
                kadayashi::kadayashi_simple_single_region_wrapper(
                        bam_reader.fp(), bam_reader.idx(), bam_reader.hdr(),
                        fastx_reader.get_raw_faidx_ptr(), "Nonexistent", 200000, 200001,
                        "" /*readgroup*/, DISABLE_INTERVAL_EXPANSION, MIN_BASE_QUALITY,
                        MIN_VARCALL_COVERAGE, MIN_VARCALL_FRACTION, MAX_CLIPPING, MIN_STRAND_COV,
                        MIN_STRAND_COV_FRAC, MAX_GAPCOMPRESSED_SEQDIV);
        CATCH_CHECK(compare_haptags(result, expected));
    }
}

CATCH_TEST_CASE("kadayashi_convert_fullinfo_var_to_dorado_style", TEST_GROUP) {
    struct TestCase {
        std::string name;
        kadayashi::variant_fullinfo_t input;
        bool keep_phasing;
        dorado::secondary::Variant expected;
    };

    const std::vector<TestCase> test_cases{
            {
                    "invalid variant returns empty Variant",
                    kadayashi::variant_fullinfo_t{.is_valid = false},
                    false,
                    dorado::secondary::Variant{},
            },
            {
                    "single allele unphased sorts genotype",
                    kadayashi::variant_fullinfo_t{.is_valid = true,
                                                  .is_confident = true,
                                                  .is_multi_allele = false,
                                                  .min_depth = 7,
                                                  .median_depth = 7,
                                                  .pos0 = 11,
                                                  .ref_allele_seq0 = "A",
                                                  .alt_allele_seq0 = "T",
                                                  .is_phased0 = true,
                                                  .genotype0 = {'1', '|', '0'}},
                    false,
                    dorado::secondary::Variant{.seq_id = -1,
                                               .pos = 11,
                                               .ref = "A",
                                               .alts = {"T"},
                                               .filter = dorado::secondary::VARIANT_PASS_FILTER,
                                               .info = {},
                                               .qual = 60.0f,
                                               .format = {{"GT", "0/1"}, {"GQ", "60"}, {"DP", "7"}},
                                               .rstart = 0,
                                               .rend = 0},
            },
            {
                    "single allele phased keeps genotype order and delimiter",
                    kadayashi::variant_fullinfo_t{.is_valid = true,
                                                  .is_confident = true,
                                                  .is_multi_allele = false,
                                                  .min_depth = 5,
                                                  .median_depth = 5,
                                                  .pos0 = 12,
                                                  .ref_allele_seq0 = "C",
                                                  .alt_allele_seq0 = "G",
                                                  .is_phased0 = true,
                                                  .genotype0 = {'1', '|', '0'}},
                    true,
                    dorado::secondary::Variant{.seq_id = -1,
                                               .pos = 12,
                                               .ref = "C",
                                               .alts = {"G"},
                                               .filter = dorado::secondary::VARIANT_PASS_FILTER,
                                               .info = {},
                                               .qual = 60.0f,
                                               .format = {{"GT", "1|0"}, {"GQ", "60"}, {"DP", "5"}},
                                               .rstart = 0,
                                               .rend = 0},
            },
            {
                    "single allele homozygous alt keeps 1/1 genotype",
                    kadayashi::variant_fullinfo_t{.is_valid = true,
                                                  .is_confident = true,
                                                  .is_multi_allele = false,
                                                  .min_depth = 6,
                                                  .median_depth = 6,
                                                  .pos0 = 17,
                                                  .ref_allele_seq0 = "G",
                                                  .alt_allele_seq0 = "A",
                                                  .is_phased0 = false,
                                                  .genotype0 = {'1', '/', '1'}},
                    false,
                    dorado::secondary::Variant{.seq_id = -1,
                                               .pos = 17,
                                               .ref = "G",
                                               .alts = {"A"},
                                               .filter = dorado::secondary::VARIANT_PASS_FILTER,
                                               .info = {},
                                               .qual = 60.0f,
                                               .format = {{"GT", "1/1"}, {"GQ", "60"}, {"DP", "6"}},
                                               .rstart = 0,
                                               .rend = 0},
            },
            {
                    "single allele low confidence keeps min depth for multi-base ref",
                    kadayashi::variant_fullinfo_t{.is_valid = true,
                                                  .is_confident = false,
                                                  .is_multi_allele = false,
                                                  .min_depth = 3,
                                                  .median_depth = 9,
                                                  .pos0 = 13,
                                                  .ref_allele_seq0 = "AC",
                                                  .alt_allele_seq0 = "A",
                                                  .is_phased0 = false,
                                                  .genotype0 = {'0', '/', '1'}},
                    true,
                    dorado::secondary::Variant{
                            .seq_id = -1,
                            .pos = 13,
                            .ref = "AC",
                            .alts = {"A"},
                            .filter = dorado::secondary::VARIANT_LOWQUAL_FILTER,
                            .info = {},
                            .qual = 0.0f,
                            .format = {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "9"}, {"MIN_DP", "3"}},
                            .rstart = 0,
                            .rend = 0},
            },
            {
                    "multi allele with same refs sorts alts and genotype when keep_phasing is "
                    "false",
                    kadayashi::variant_fullinfo_t{.is_valid = true,
                                                  .is_confident = true,
                                                  .is_multi_allele = true,
                                                  .min_depth = 8,
                                                  .median_depth = 8,
                                                  .pos0 = 14,
                                                  .ref_allele_seq0 = "A",
                                                  .alt_allele_seq0 = "T",
                                                  .is_phased0 = true,
                                                  .genotype0 = {'1', '|', '0'},
                                                  .ref_allele_seq1 = "A",
                                                  .alt_allele_seq1 = "C"},
                    false,
                    dorado::secondary::Variant{.seq_id = -1,
                                               .pos = 14,
                                               .ref = "A",
                                               .alts = {"C", "T"},
                                               .filter = dorado::secondary::VARIANT_PASS_FILTER,
                                               .info = {},
                                               .qual = 60.0f,
                                               .format = {{"GT", "1/2"}, {"GQ", "60"}, {"DP", "8"}},
                                               .rstart = 0,
                                               .rend = 0},
            },
            {
                    "multi allele with same refs sorts alts when keep_phasing is false",
                    kadayashi::variant_fullinfo_t{.is_valid = true,
                                                  .is_confident = true,
                                                  .is_multi_allele = true,
                                                  .min_depth = 8,
                                                  .median_depth = 8,
                                                  .pos0 = 14,
                                                  .ref_allele_seq0 = "A",
                                                  .alt_allele_seq0 = "T",
                                                  .is_phased0 = true,
                                                  .genotype0 = {'1', '|', '0'},
                                                  .ref_allele_seq1 = "A",
                                                  .alt_allele_seq1 = "C"},
                    false,
                    dorado::secondary::Variant{.seq_id = -1,
                                               .pos = 14,
                                               .ref = "A",
                                               .alts = {"C", "T"},
                                               .filter = dorado::secondary::VARIANT_PASS_FILTER,
                                               .info = {},
                                               .qual = 60.0f,
                                               .format = {{"GT", "1/2"}, {"GQ", "60"}, {"DP", "8"}},
                                               .rstart = 0,
                                               .rend = 0},
            },
            {
                    "multi allele with same refs sorts genotype when keep_phasing is false",
                    kadayashi::variant_fullinfo_t{.is_valid = true,
                                                  .is_confident = true,
                                                  .is_multi_allele = true,
                                                  .min_depth = 8,
                                                  .median_depth = 8,
                                                  .pos0 = 14,
                                                  .ref_allele_seq0 = "A",
                                                  .alt_allele_seq0 = "C",
                                                  .is_phased0 = true,
                                                  .genotype0 = {'0', '|', '1'},
                                                  .ref_allele_seq1 = "A",
                                                  .alt_allele_seq1 = "T"},
                    false,
                    dorado::secondary::Variant{.seq_id = -1,
                                               .pos = 14,
                                               .ref = "A",
                                               .alts = {"C", "T"},
                                               .filter = dorado::secondary::VARIANT_PASS_FILTER,
                                               .info = {},
                                               .qual = 60.0f,
                                               .format = {{"GT", "1/2"}, {"GQ", "60"}, {"DP", "8"}},
                                               .rstart = 0,
                                               .rend = 0},
            },
            {
                    "multi allele with same refs keeps alt and genotype order when keep_phasing is "
                    "true",
                    kadayashi::variant_fullinfo_t{.is_valid = true,
                                                  .is_confident = true,
                                                  .is_multi_allele = true,
                                                  .min_depth = 8,
                                                  .median_depth = 8,
                                                  .pos0 = 14,
                                                  .ref_allele_seq0 = "A",
                                                  .alt_allele_seq0 = "T",
                                                  .is_phased0 = true,
                                                  .genotype0 = {'0', '|', '1'},
                                                  .ref_allele_seq1 = "A",
                                                  .alt_allele_seq1 = "C"},
                    true,
                    dorado::secondary::Variant{.seq_id = -1,
                                               .pos = 14,
                                               .ref = "A",
                                               .alts = {"T", "C"},
                                               .filter = dorado::secondary::VARIANT_PASS_FILTER,
                                               .info = {},
                                               .qual = 60.0f,
                                               .format = {{"GT", "2|1"}, {"GQ", "60"}, {"DP", "8"}},
                                               .rstart = 0,
                                               .rend = 0},
            },
            {
                    "multi allele with longer first ref extends second alt",
                    kadayashi::variant_fullinfo_t{.is_valid = true,
                                                  .is_confident = true,
                                                  .is_multi_allele = true,
                                                  .min_depth = 4,
                                                  .median_depth = 10,
                                                  .pos0 = 15,
                                                  .ref_allele_seq0 = "ACG",
                                                  .alt_allele_seq0 = "T",
                                                  .is_phased0 = true,
                                                  .genotype0 = {'0', '|', '1'},
                                                  .ref_allele_seq1 = "A",
                                                  .alt_allele_seq1 = "T"},
                    true,
                    dorado::secondary::Variant{
                            .seq_id = -1,
                            .pos = 15,
                            .ref = "ACG",
                            .alts = {"T", "TCG"},
                            .filter = dorado::secondary::VARIANT_PASS_FILTER,
                            .info = {},
                            .qual = 60.0f,
                            .format = {{"GT", "2|1"}, {"GQ", "60"}, {"DP", "10"}, {"MIN_DP", "4"}},
                            .rstart = 0,
                            .rend = 0},
            },
            {
                    "multi allele with longer second ref extends first alt",
                    kadayashi::variant_fullinfo_t{.is_valid = true,
                                                  .is_confident = true,
                                                  .is_multi_allele = true,
                                                  .min_depth = 6,
                                                  .median_depth = 6,
                                                  .pos0 = 16,
                                                  .ref_allele_seq0 = "A",
                                                  .alt_allele_seq0 = "C",
                                                  .is_phased0 = false,
                                                  .genotype0 = {'1', '/', '0'},
                                                  .ref_allele_seq1 = "ACC",
                                                  .alt_allele_seq1 = "T"},
                    false,
                    dorado::secondary::Variant{.seq_id = -1,
                                               .pos = 16,
                                               .ref = "ACC",
                                               .alts = {"CCC", "T"},
                                               .filter = dorado::secondary::VARIANT_PASS_FILTER,
                                               .info = {},
                                               .qual = 60.0f,
                                               .format = {{"GT", "1/2"}, {"GQ", "60"}, {"DP", "6"}},
                                               .rstart = 0,
                                               .rend = 0},
            },
            {
                    // Regression test for a het deletion where one haplotype matches the reference.
                    // The reference-matching haplotype is encoded as an ALT that, after ref-length
                    // normalization, becomes identical to REF. It must be collapsed into genotype 0
                    // rather than emitted as a redundant ALT duplicating REF (previously produced
                    // REF=ATATCTTATCT ALT=A,ATATCTTATCT GT=1/2).
                    "multi allele with a reference-matching haplotype collapses to a het call",
                    kadayashi::variant_fullinfo_t{.is_valid = true,
                                                  .is_confident = true,
                                                  .is_multi_allele = true,
                                                  .min_depth = 23,
                                                  .median_depth = 23,
                                                  .pos0 = 10378832,
                                                  .ref_allele_seq0 = "ATATCTTATCT",
                                                  .alt_allele_seq0 = "A",
                                                  .is_phased0 = false,
                                                  .genotype0 = {'1', '/', '0'},
                                                  .ref_allele_seq1 = "A",
                                                  .alt_allele_seq1 = "A"},
                    false,
                    dorado::secondary::Variant{
                            .seq_id = -1,
                            .pos = 10378832,
                            .ref = "ATATCTTATCT",
                            .alts = {"A"},
                            .filter = dorado::secondary::VARIANT_PASS_FILTER,
                            .info = {},
                            .qual = 60.0f,
                            .format = {{"GT", "0/1"}, {"GQ", "60"}, {"DP", "23"}},
                            .rstart = 0,
                            .rend = 0},
            },
    };

    for (const auto &test_case : test_cases) {
        CATCH_CAPTURE(test_case.name);
        const dorado::secondary::Variant result = kadayashi::convert_fullinfo_var_to_dorado_style(
                test_case.input, test_case.keep_phasing);
        CATCH_CHECK(result == test_case.expected);
    }
}

CATCH_TEST_CASE("kadayashi_varcall normal case", TEST_GROUP) {
    // Input data.
    const std::filesystem::path test_data_dir = get_data_dir("variant") / "test-02-supertiny";
    const std::filesystem::path fn_bam = test_data_dir / "in.aln.bam";
    const std::filesystem::path fn_ref = test_data_dir / "in.ref.fasta.gz";

    const kadayashi::varcall_result_t expected{
            .qname2hp =
                    {
                            {"1e70cda3-c41f-4d19-9c14-94d8d64e619c", 1},
                            {"61ab09d6-072f-4ab2-b14b-b0a1e38a3419", 1},
                            {"563ecca1-30dd-4dd9-991a-d417d827c803", 0},
                            {"4fd81aa2-cb77-4994-a8a5-70e6228f255e", 0},
                            {"a27cad27-2297-40d4-8666-40a4742eb2ed", 1},
                            {"e0af6c87-8655-4603-97b7-0ad5ba860df2", 0},
                            {"7d23577c-5c93-4d41-83bd-b652e687deee", 0},
                            {"627ea9e1-5204-4a2c-ae54-1e1be8bbbbe6", 1},
                            {"ac863a7d-932e-42fa-91c1-7814d7f810f9", 1},
                            {"b4139858-e420-4780-94e6-375542c2d2e8", 0},
                            {"dbe9785a-fa25-454c-9960-fd65fb99a040", 1},
                            {"3fdc1b9b-7186-411e-af92-e93a1086754c", 1},
                            {"7b2095d4-08f7-448d-aa9d-55c9568fb49d", 1},
                            {"c488f4c5-1639-4be1-92f6-948f29b7d822", 1},
                            {"02551418-20c9-4b4b-9d1b-9bee36342895", 1},
                            {"de45db56-e704-4524-af88-06a2f98c270e", 1},
                            {"49b05d0d-97ac-449e-804b-35b35e05ce28", 0},
                            {"e7e27cb5-1144-49dd-8ec4-09a75937a091", 0},
                            {"3d7a9813-67be-4b84-b66a-0269aa108340", 1},
                            {"d5560893-59c8-417c-a929-d62b4d19a1ca", 1},
                    },
            .variants =
                    {
                            // clang-format off
                            // 0-index
                            {-1, 93, "C", {"T"}, dorado::secondary::VARIANT_PASS_FILTER, {}, 60.0f, {{"GT", "0/1"}, {"GQ", "60"}, {"DP", "20"}}},
                            {-1, 305, "G", {"A"}, dorado::secondary::VARIANT_PASS_FILTER, {}, 60.0f, {{"GT", "0/1"}, {"GQ", "60"}, {"DP", "20"}}},
                            {-1, 775, "TC", {"T"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "20"}}},
                            {-1, 809, "AC", {"A"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "20"}}},
                            {-1, 1002, "AC", {"A"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "20"}}},
                            {-1, 1471, "T", {"G"}, dorado::secondary::VARIANT_PASS_FILTER, {}, 60.0f, {{"GT", "0/1"}, {"GQ", "60"}, {"DP", "20"}}},
                            {-1, 1613, "TC", {"T"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "20"}}},
                            {-1, 1619, "CG", {"C"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "20"}}},
                            {-1, 1829, "CTTT", {"C"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "20"}}},
                            {-1, 1958, "T", {"G"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "19"}}},
                            {-1, 2101, "A", {"G"}, dorado::secondary::VARIANT_PASS_FILTER, {}, 60.0f, {{"GT", "0/1"}, {"GQ", "60"}, {"DP", "19"}}},
                            {-1, 2125, "C", {"G"}, dorado::secondary::VARIANT_PASS_FILTER, {}, 60.0f, {{"GT", "0/1"}, {"GQ", "60"}, {"DP", "19"}}},
                            {-1, 2437, "T", {"C"}, dorado::secondary::VARIANT_PASS_FILTER, {}, 60.0f, {{"GT", "0/1"}, {"GQ", "60"}, {"DP", "19"}}},
                            {-1, 2445, "G", {"A"}, dorado::secondary::VARIANT_PASS_FILTER, {}, 60.0f, {{"GT", "0/1"}, {"GQ", "60"}, {"DP", "19"}}},
                            {-1, 2966, "G", {"C"}, dorado::secondary::VARIANT_PASS_FILTER, {}, 60.0f, {{"GT", "1/1"}, {"GQ", "60"}, {"DP", "18"}}},
                            {-1, 3175, "G", {"T"}, dorado::secondary::VARIANT_PASS_FILTER, {}, 60.0f, {{"GT", "0/1"}, {"GQ", "60"}, {"DP", "18"}}},
                            {-1, 3514, "CT", {"C"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "18"}}},
                            {-1, 3734, "G", {"C"}, dorado::secondary::VARIANT_PASS_FILTER, {}, 60.0f, {{"GT", "1/1"}, {"GQ", "60"}, {"DP", "18"}}},
                            {-1, 3961, "A", {"AC"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "18"}}},
                            {-1, 4695, "A", {"T"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "18"}}},
                            {-1, 4999, "AAAAT", {"A", "AAAATAAAT"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "1/2"}, {"GQ", "0"}, {"DP", "18"}}},
                            {-1, 5125, "CTTT", {"C"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "18"}}},
                            {-1, 5386, "C", {"G"}, dorado::secondary::VARIANT_PASS_FILTER, {}, 60.0f, {{"GT", "0/1"}, {"GQ", "60"}, {"DP", "17"}}},
                            {-1, 5985, "CA", {"C"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "17"}}},
                            {-1, 6947, "CGTGT", {"C"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "15"}}},
                            {-1, 7429, "C", {"T"}, dorado::secondary::VARIANT_PASS_FILTER, {}, 60.0f, {{"GT", "0/1"}, {"GQ", "60"}, {"DP", "15"}}},
                            {-1, 7690, "TCC", {"T"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "15"}}},
                            {-1, 7912, "C", {"CT"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "15"}}},
                            {-1, 8319, "GAA", {"G"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "15"}}},
                            // clang-format on
                    },
            .read_endpoints = {{0, 1897},  {0, 2452},  {0, 5143},  {0, 6234},  {0, 6518},
                               {0, 10000}, {0, 10000}, {0, 10000}, {0, 10000}, {0, 10000},
                               {0, 10000}, {0, 10000}, {0, 10000}, {0, 10000}, {0, 10000},
                               {0, 10000}, {0, 10000}, {0, 10000}, {0, 10000}, {0, 10000}}};

    // Open the input files.
    dorado::secondary::BamFile bam_reader(fn_bam, 1);
    dorado::hts_io::FastxRandomReader fastx_reader(fn_ref);

    CATCH_REQUIRE(bam_reader.fp());
    CATCH_REQUIRE(bam_reader.idx());
    CATCH_REQUIRE(bam_reader.hdr());
    CATCH_REQUIRE(fastx_reader.get_raw_faidx_ptr());

    const kadayashi::pileup_pars_t pp{.max_clipping = 100000};

    CATCH_SECTION("simple phasing varcall") {
        const kadayashi::varcall_result_t result = kadayashi::kadayashi_phase_and_varcall_wrapper(
                bam_reader.fp(), bam_reader.idx(), bam_reader.hdr(),
                fastx_reader.get_raw_faidx_ptr(), "chr20", 0, 10000, "" /*readgroup*/,
                pp.disable_region_expansion, pp.min_base_quality, pp.min_varcall_coverage,
                pp.min_varcall_fraction, pp.max_clipping, 1 /*min strand cov*/, 0.033f,
                pp.max_gapcompressed_seqdiv, false /*ambig_ref*/);
        CATCH_CHECK(compare_haptags(result.qname2hp, expected.qname2hp));
        CATCH_CHECK(result.variants == expected.variants);
        CATCH_CHECK(result.read_endpoints == expected.read_endpoints);
    }

    // Check that the first variant is returned correctly if the chunk starts on a variant
    CATCH_SECTION("simple phasing varcall, variant at chunk start") {
        const kadayashi::varcall_result_t result = kadayashi::kadayashi_phase_and_varcall_wrapper(
                bam_reader.fp(), bam_reader.idx(), bam_reader.hdr(),
                fastx_reader.get_raw_faidx_ptr(), "chr20", 93, 10000, "" /*readgroup*/,
                pp.disable_region_expansion, pp.min_base_quality, pp.min_varcall_coverage,
                pp.min_varcall_fraction, pp.max_clipping, 1 /*min strand cov*/, 0.033f,
                pp.max_gapcompressed_seqdiv, false /*ambig_ref*/);
        CATCH_CHECK(compare_haptags(result.qname2hp, expected.qname2hp));
        CATCH_CHECK(result.variants == expected.variants);
        {
            std::vector<std::pair<uint32_t, uint32_t>> this_endpoints;
            this_endpoints.reserve(std::size(expected.read_endpoints));
            for (const auto &pair : expected.read_endpoints) {
                this_endpoints.emplace_back(std::make_pair<uint32_t>(93, pair.second));
            }
            CATCH_CHECK(result.read_endpoints == this_endpoints);
        }
    }

    // Check that the last variant is returned correctly if the chunk ends on a variant
    // The expected behaviour is that if a deletion spans the end of the chunk, the entire
    // deletion is reported
    CATCH_SECTION("simple phasing varcall, variant at chunk end") {
        const kadayashi::varcall_result_t result = kadayashi::kadayashi_phase_and_varcall_wrapper(
                bam_reader.fp(), bam_reader.idx(), bam_reader.hdr(),
                fastx_reader.get_raw_faidx_ptr(), "chr20", 0, 8321 /*end-inclusive*/,
                "" /*readgroup*/, pp.disable_region_expansion, pp.min_base_quality,
                pp.min_varcall_coverage, pp.min_varcall_fraction, pp.max_clipping,
                1 /*min strand cov*/, 0.033f, pp.max_gapcompressed_seqdiv, false /*ambig_ref*/);
        CATCH_CHECK(compare_haptags(result.qname2hp, expected.qname2hp));
        CATCH_CHECK(result.variants == expected.variants);
        {
            std::vector<std::pair<uint32_t, uint32_t>> this_endpoints;
            this_endpoints.reserve(std::size(expected.read_endpoints));
            for (const auto &pair : expected.read_endpoints) {
                this_endpoints.emplace_back(
                        std::make_pair(0, std::min<uint32_t>(pair.second, 8321)));
            }
            CATCH_CHECK(result.read_endpoints == this_endpoints);
        }
    }

    CATCH_SECTION("use wrong clipping threshold") {
        const kadayashi::varcall_result_t result3 = kadayashi::kadayashi_phase_and_varcall_wrapper(
                bam_reader.fp(), bam_reader.idx(), bam_reader.hdr(),
                fastx_reader.get_raw_faidx_ptr(), "chr20", 0, 9999, "" /*readgroup*/,
                pp.disable_region_expansion, pp.min_base_quality, pp.min_varcall_coverage,
                pp.min_varcall_fraction, 100, 1 /*min strand cov*/, 0.033f,
                pp.max_gapcompressed_seqdiv, false /*ambig_ref*/);
        CATCH_CHECK(result3.variants.empty());
        CATCH_CHECK(result3.read_endpoints.empty());
    }
}
CATCH_TEST_CASE("kadayashi_varcall normal case2", TEST_GROUP) {
    // The DEL should be phased and be a multi-allele DEL due to
    // taking in a SNP on the other hap.
    const std::filesystem::path test_data_dir =
            get_data_dir("variant") / "test-08-supertiny-deletion-absorbtion";
    const std::filesystem::path fn_bam = test_data_dir / "in.aln.bam";
    const std::filesystem::path fn_ref = test_data_dir / "in.ref.fa.gz";

    dorado::secondary::BamFile bam_reader(fn_bam, 1);
    dorado::hts_io::FastxRandomReader fastx_reader(fn_ref);

    CATCH_REQUIRE(bam_reader.fp());
    CATCH_REQUIRE(bam_reader.idx());
    CATCH_REQUIRE(bam_reader.hdr());
    CATCH_REQUIRE(fastx_reader.get_raw_faidx_ptr());

    const kadayashi::pileup_pars_t pp{.max_clipping = 100000};

    const kadayashi::varcall_result_t result = kadayashi::kadayashi_phase_and_varcall_wrapper(
            bam_reader.fp(), bam_reader.idx(), bam_reader.hdr(), fastx_reader.get_raw_faidx_ptr(),
            "chr1:23935672-23936360", 0, 700, "" /*readgroup*/, pp.disable_region_expansion,
            pp.min_base_quality, pp.min_varcall_coverage, pp.min_varcall_fraction, pp.max_clipping,
            1 /*min strand cov*/, 0.033f, pp.max_gapcompressed_seqdiv, false /*ambig_ref*/);

    int passed = 0;
    for (const dorado::secondary::Variant &var : result.variants) {
        if (var.pos == 113) {  // coordinate on hg38: 23935784 (0-idx)
            CATCH_REQUIRE(var.filter == dorado::secondary::VARIANT_PASS_FILTER);
            CATCH_REQUIRE(var.format.at("GT") == "1/2");
            CATCH_REQUIRE(var.alts.size() == 2);
            CATCH_REQUIRE(!var.alts[0].empty());
            CATCH_REQUIRE(!var.alts[1].empty());
            passed++;
        }
    }
    CATCH_REQUIRE(passed == 1);
}

CATCH_TEST_CASE("kadayashi_varcall ambig-ref", TEST_GROUP) {
    // Check that the ambig-ref argument correctly allows variants to be returned at
    // positions with ambiguous reference bases, and that they are not returned if the
    // argument is not set.

    // Input data.
    const std::filesystem::path test_data_dir =
            get_data_dir("variant") / "test-05-kadayashi-varcall-ambig-ref";
    const std::filesystem::path fn_bam = test_data_dir / "in.aln.bam";
    const std::filesystem::path fn_ref = test_data_dir / "in.ref.fasta.gz";

    // Expected output when ambig_ref is true.
    const kadayashi::varcall_result_t expected_ambig_ref_true{
            .qname2hp =
                    {
                            {"1e70cda3-c41f-4d19-9c14-94d8d64e619c", 1},
                            {"61ab09d6-072f-4ab2-b14b-b0a1e38a3419", 1},
                            {"563ecca1-30dd-4dd9-991a-d417d827c803", 0},
                            {"4fd81aa2-cb77-4994-a8a5-70e6228f255e", 0},
                            {"a27cad27-2297-40d4-8666-40a4742eb2ed", 1},
                            {"e0af6c87-8655-4603-97b7-0ad5ba860df2", 0},
                            {"7d23577c-5c93-4d41-83bd-b652e687deee", 0},
                            {"627ea9e1-5204-4a2c-ae54-1e1be8bbbbe6", 1},
                            {"ac863a7d-932e-42fa-91c1-7814d7f810f9", 1},
                            {"b4139858-e420-4780-94e6-375542c2d2e8", 0},
                            {"dbe9785a-fa25-454c-9960-fd65fb99a040", 1},
                            {"3fdc1b9b-7186-411e-af92-e93a1086754c", 1},
                            {"7b2095d4-08f7-448d-aa9d-55c9568fb49d", 1},
                            {"c488f4c5-1639-4be1-92f6-948f29b7d822", 1},
                            {"02551418-20c9-4b4b-9d1b-9bee36342895", 1},
                            {"de45db56-e704-4524-af88-06a2f98c270e", 1},
                            {"49b05d0d-97ac-449e-804b-35b35e05ce28", 0},
                            {"e7e27cb5-1144-49dd-8ec4-09a75937a091", 0},
                            {"3d7a9813-67be-4b84-b66a-0269aa108340", 1},
                            {"d5560893-59c8-417c-a929-d62b4d19a1ca", 1},
                    },
            .variants = {
                    // clang-format off
                    // 0-index
                    {-1, 93, "C", {"T"}, dorado::secondary::VARIANT_PASS_FILTER, {}, 60.0f, {{"GT", "0/1"}, {"GQ", "60"}, {"DP", "20"}}},
                    {-1, 305, "Y", {"A", "G"}, dorado::secondary::VARIANT_PASS_FILTER, {}, 60.0f, {{"GT", "1/2"}, {"GQ", "60"}, {"DP", "20"}}},
                    {-1, 775, "TC", {"T"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "20"}}},
                    //unconfident variant is not consolidated
                    {-1, 809, "AN", {"A"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "20"}}},
                    {-1, 810, "S", {"C"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "20"}}},
                    //
                    {-1, 1002, "AC", {"A"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "20"}}},
                    {-1, 1471, "T", {"G"}, dorado::secondary::VARIANT_PASS_FILTER, {}, 60.0f, {{"GT", "0/1"}, {"GQ", "60"}, {"DP", "20"}}},
                    {-1, 1613, "TC", {"T"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "20"}}},
                    {-1, 1619, "CG", {"C"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "20"}}},
                    {-1, 1829, "CTTT", {"C"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "20"}}},
                    {-1, 1958, "T", {"G"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "19"}}},
                    {-1, 2101, "A", {"G"}, dorado::secondary::VARIANT_PASS_FILTER, {}, 60.0f, {{"GT", "0/1"}, {"GQ", "60"}, {"DP", "19"}}},
                    {-1, 2125, "C", {"G"}, dorado::secondary::VARIANT_PASS_FILTER, {}, 60.0f, {{"GT", "0/1"}, {"GQ", "60"}, {"DP", "19"}}},
                    {-1, 2437, "T", {"C"}, dorado::secondary::VARIANT_PASS_FILTER, {}, 60.0f, {{"GT", "0/1"}, {"GQ", "60"}, {"DP", "19"}}},
                    {-1, 2445, "B", {"A", "G"}, dorado::secondary::VARIANT_PASS_FILTER, {}, 60.0f, {{"GT", "1/2"}, {"GQ", "60"}, {"DP", "19"}}},
                    {-1, 2966, "G", {"C"}, dorado::secondary::VARIANT_PASS_FILTER, {}, 60.0f, {{"GT", "1/1"}, {"GQ", "60"}, {"DP", "18"}}},
                    {-1, 3175, "G", {"T"}, dorado::secondary::VARIANT_PASS_FILTER, {}, 60.0f, {{"GT", "0/1"}, {"GQ", "60"}, {"DP", "18"}}},
                    {-1, 3514, "CT", {"C"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "18"}}},
                    {-1, 3734, "G", {"C"}, dorado::secondary::VARIANT_PASS_FILTER, {}, 60.0f, {{"GT", "1/1"}, {"GQ", "60"}, {"DP", "18"}}},
                    {-1, 3961, "A", {"AC"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "18"}}},
                    {-1, 4695, "A", {"T"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "18"}}},
                    //unconfident variant is not consolidated
                    {-1, 4999, "ANNNT", {"A", "AAAATNNNT"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "1/2"}, {"GQ", "0"}, {"DP", "18"}}},
                    {-1, 5000, "K", {"A"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "18"}}},
                    {-1, 5001, "K", {"A"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "18"}}},
                    {-1, 5002, "K", {"A"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "18"}}},
                    //
                    {-1, 5125, "CTTT", {"C"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "18"}}},
                    {-1, 5386, "C", {"G"}, dorado::secondary::VARIANT_PASS_FILTER, {}, 60.0f, {{"GT", "0/1"}, {"GQ", "60"}, {"DP", "17"}}},
                    {-1, 5985, "CA", {"C"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "17"}}},
                    {-1, 6947, "CGTGT", {"C"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "15"}}},
                    {-1, 7429, "C", {"T"}, dorado::secondary::VARIANT_PASS_FILTER, {}, 60.0f, {{"GT", "0/1"}, {"GQ", "60"}, {"DP", "15"}}},
                    {-1, 7690, "TCC", {"T"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "15"}}},
                    {-1, 7912, "C", {"CT"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "15"}}},
                    {-1, 8319, "GAA", {"G"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "15"}}},
                    // clang-format on
            }};

    // Expected output when ambig_ref is false.
    const kadayashi::varcall_result_t expected_ambig_ref_false{
            .qname2hp =
                    {
                            {"1e70cda3-c41f-4d19-9c14-94d8d64e619c", 1},
                            {"61ab09d6-072f-4ab2-b14b-b0a1e38a3419", 1},
                            {"563ecca1-30dd-4dd9-991a-d417d827c803", 0},
                            {"4fd81aa2-cb77-4994-a8a5-70e6228f255e", 0},
                            {"a27cad27-2297-40d4-8666-40a4742eb2ed", 1},
                            {"e0af6c87-8655-4603-97b7-0ad5ba860df2", 0},
                            {"7d23577c-5c93-4d41-83bd-b652e687deee", 0},
                            {"627ea9e1-5204-4a2c-ae54-1e1be8bbbbe6", 1},
                            {"ac863a7d-932e-42fa-91c1-7814d7f810f9", 1},
                            {"b4139858-e420-4780-94e6-375542c2d2e8", 0},
                            {"dbe9785a-fa25-454c-9960-fd65fb99a040", 1},
                            {"3fdc1b9b-7186-411e-af92-e93a1086754c", 1},
                            {"7b2095d4-08f7-448d-aa9d-55c9568fb49d", 1},
                            {"c488f4c5-1639-4be1-92f6-948f29b7d822", 1},
                            {"02551418-20c9-4b4b-9d1b-9bee36342895", 1},
                            {"de45db56-e704-4524-af88-06a2f98c270e", 1},
                            {"49b05d0d-97ac-449e-804b-35b35e05ce28", 0},
                            {"e7e27cb5-1144-49dd-8ec4-09a75937a091", 0},
                            {"3d7a9813-67be-4b84-b66a-0269aa108340", 1},
                            {"d5560893-59c8-417c-a929-d62b4d19a1ca", 1},
                    },
            .variants = {
                    // clang-format off
                    // 0-index
                    // some genotypes are flipped wrt above
                    {-1, 93, "C", {"T"}, dorado::secondary::VARIANT_PASS_FILTER, {}, 60.0f, {{"GT", "0/1"}, {"GQ", "60"}, {"DP", "20"}}},
                    // {true, true, 305, "Y", {"A", "G"}, {{"GT", "1/2"}, {"DP", "20"}}},
                    {-1, 775, "TC", {"T"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "20"}}},
                    //unconfident variant is not consolidated
                    // {-1, 809, "AN", {"A"}, {{"GT", "0/1"}, {"DP", "20"}}},
                    // {-1, 810, "S", {"C"}, {{"GT", "0/1"}, {"DP", "20"}}},
                    //
                    {-1, 1002, "AC", {"A"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "20"}}},
                    {-1, 1471, "T", {"G"}, dorado::secondary::VARIANT_PASS_FILTER, {}, 60.0f, {{"GT", "0/1"}, {"GQ", "60"}, {"DP", "20"}}},
                    {-1, 1613, "TC", {"T"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "20"}}},
                    {-1, 1619, "CG", {"C"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "20"}}},
                    {-1, 1829, "CTTT", {"C"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "20"}}},
                    {-1, 1958, "T", {"G"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "19"}}},
                    {-1, 2101, "A", {"G"}, dorado::secondary::VARIANT_PASS_FILTER, {}, 60.0f, {{"GT", "0/1"}, {"GQ", "60"}, {"DP", "19"}}},
                    {-1, 2125, "C", {"G"}, dorado::secondary::VARIANT_PASS_FILTER, {}, 60.0f, {{"GT", "0/1"}, {"GQ", "60"}, {"DP", "19"}}},
                    {-1, 2437, "T", {"C"}, dorado::secondary::VARIANT_PASS_FILTER, {}, 60.0f, {{"GT", "0/1"}, {"GQ", "60"}, {"DP", "19"}}},
                    // {-1, 2445, "B", {"A", "G"}, dorado::secondary::VARIANT_PASS_FILTER, {}, 60.0f, {{"GT", "1/2"}, {"GQ", "60"}, {"DP", "19"}}},
                    {-1, 2966, "G", {"C"}, dorado::secondary::VARIANT_PASS_FILTER, {}, 60.0f, {{"GT", "1/1"}, {"GQ", "60"}, {"DP", "18"}}},
                    {-1, 3175, "G", {"T"}, dorado::secondary::VARIANT_PASS_FILTER, {}, 60.0f, {{"GT", "0/1"}, {"GQ", "60"}, {"DP", "18"}}},
                    {-1, 3514, "CT", {"C"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "18"}}},
                    {-1, 3734, "G", {"C"}, dorado::secondary::VARIANT_PASS_FILTER, {}, 60.0f, {{"GT", "1/1"}, {"GQ", "60"}, {"DP", "18"}}},
                    {-1, 3961, "A", {"AC"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "18"}}},
                    {-1, 4695, "A", {"T"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "18"}}},
                    //unconfident variant is not consolidated
                    // {-1, 4999, "ANNNT", {"A", "AAAATNNNT"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "1/2"}, {"GQ", "0"}, {"DP", "18"}}},
                    // {-1, 5000, "K", {"A"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "18"}}},
                    // {-1, 5001, "K", {"A"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "18"}}},
                    // {-1, 5002, "K", {"A"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "18"}}},
                    //
                    {-1, 5125, "CTTT", {"C"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "18"}}},
                    {-1, 5386, "C", {"G"}, dorado::secondary::VARIANT_PASS_FILTER, {}, 60.0f, {{"GT", "0/1"}, {"GQ", "60"}, {"DP", "17"}}},
                    {-1, 5985, "CA", {"C"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "17"}}},
                    {-1, 6947, "CGTGT", {"C"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "15"}}},
                    {-1, 7429, "C", {"T"}, dorado::secondary::VARIANT_PASS_FILTER, {}, 60.0f, {{"GT", "0/1"}, {"GQ", "60"}, {"DP", "15"}}},
                    {-1, 7690, "TCC", {"T"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "15"}}},
                    {-1, 7912, "C", {"CT"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "15"}}},
                    {-1, 8319, "GAA", {"G"}, dorado::secondary::VARIANT_LOWQUAL_FILTER, {}, 0.0f, {{"GT", "0/1"}, {"GQ", "0"}, {"DP", "15"}}},
                    // clang-format on
            }};

    // Open the input files.
    dorado::secondary::BamFile bam_reader(fn_bam, 1);
    dorado::hts_io::FastxRandomReader fastx_reader(fn_ref);

    CATCH_REQUIRE(bam_reader.fp());
    CATCH_REQUIRE(bam_reader.idx());
    CATCH_REQUIRE(bam_reader.hdr());
    CATCH_REQUIRE(fastx_reader.get_raw_faidx_ptr());

    const kadayashi::pileup_pars_t pp{.max_clipping = 100000};

    CATCH_SECTION("ambig-ref true") {
        const kadayashi::varcall_result_t result = kadayashi::kadayashi_phase_and_varcall_wrapper(
                bam_reader.fp(), bam_reader.idx(), bam_reader.hdr(),
                fastx_reader.get_raw_faidx_ptr(), "chr20", 0, 9999, "" /*readgroup*/,
                pp.disable_region_expansion, pp.min_base_quality, pp.min_varcall_coverage,
                pp.min_varcall_fraction, pp.max_clipping, 1 /*min strand cov*/, 0.033f,
                pp.max_gapcompressed_seqdiv, true);
        CATCH_CHECK(compare_haptags(result.qname2hp, expected_ambig_ref_true.qname2hp));
        CATCH_CHECK(result.variants == expected_ambig_ref_true.variants);
    }

    CATCH_SECTION("ambig-ref false") {
        const kadayashi::varcall_result_t result = kadayashi::kadayashi_phase_and_varcall_wrapper(
                bam_reader.fp(), bam_reader.idx(), bam_reader.hdr(),
                fastx_reader.get_raw_faidx_ptr(), "chr20", 0, 9999, "" /*readgroup*/,
                pp.disable_region_expansion, pp.min_base_quality, pp.min_varcall_coverage,
                pp.min_varcall_fraction, pp.max_clipping, 1 /*min strand cov*/, 0.033f,
                pp.max_gapcompressed_seqdiv, false);
        CATCH_CHECK(compare_haptags(result.qname2hp, expected_ambig_ref_false.qname2hp));
        CATCH_CHECK(result.variants == expected_ambig_ref_false.variants);
    }
}

CATCH_TEST_CASE("kadayashi_featmatgen normal case", TEST_GROUP) {
    kadayashi::MedakaFeatureMatrixOptions medaka_feature_matrix_options = {
            .include_dwells = true,
            .include_haplotype_column = true,
            .include_snp_qv = true,
            .min_mapq = 1,
            .num_dtypes = 1,
            .dtypes = {},
            .tag_name = "",
            .tag_value = 0,
            .tag_keep_missing = false,
            .readgroup = "",
            .disable_read_packing = false,
            .hap_source = kadayashi::USE_TAG_FROM_HASHTABLE,
            .max_reads = 100,
            .right_align_insertions = true,
            .min_snp_accuracy = 0.0};
    kadayashi::str2int_t qname2hp;

    const std::filesystem::path test_data_dir =
            get_data_dir("variant") / "test-04-kadayashi-featmatgen";
    const std::filesystem::path fn_bam = test_data_dir / "in_aln_chr20_10M_10M1k.bam";
    const std::filesystem::path fn_out_expected =
            test_data_dir / "outexpected_chr20_10M_10M1k.mfmdump.tsv";

    dorado::secondary::BamFile bam_file(fn_bam, 1);
    CATCH_REQUIRE(bam_file.fp());
    CATCH_REQUIRE(bam_file.idx());
    CATCH_REQUIRE(bam_file.hdr());

    const std::string ref_name = "ref";
    const uint32_t ref_start = 0;
    const uint32_t ref_end = 1000;
    const dorado::secondary::ReadAlignmentData result =
            kadayashi::gen_medaka_feature_matrix_wrapper(bam_file, ref_name, ref_start, ref_end, {},
                                                         medaka_feature_matrix_options);
    const std::string medaka_feature_matrix_string = kadayashi::print_medaka_feature_matrix(result);

    std::string medaka_feature_matrix_string_expected;
    std::ifstream fp_out_expected(fn_out_expected);
    std::string line;
    while (std::getline(fp_out_expected, line)) {
        medaka_feature_matrix_string_expected += line;
        medaka_feature_matrix_string_expected += '\n';
    }

    CATCH_CHECK(medaka_feature_matrix_string == medaka_feature_matrix_string_expected);
}

CATCH_TEST_CASE("kadayashi_featmatgen matches calculate_read_alignment on real data", TEST_GROUP) {
    /*
        chr20: |--------------------------- 10 kbp ---------------------------|
        reads: |==================== 20 BAM HP/mv-tagged reads ==============|

        The old generator may keep a larger backing buffer, so compare the logical matrix:
        [position, read lane, feature].
    */
    const std::filesystem::path test_data_dir = get_data_dir("variant") / "test-02-supertiny";
    const std::filesystem::path fn_bam = test_data_dir / "in.aln.bam";
    const std::filesystem::path fn_ref = test_data_dir / "in.ref.fasta.gz";

    dorado::hts_io::FastxRandomReader fastx_reader(fn_ref);
    CATCH_REQUIRE(fastx_reader.get_raw_faidx_ptr());

    const std::string ref_name = "chr20";
    const uint32_t ref_start = 0;
    const uint32_t ref_end = static_cast<uint32_t>(fastx_reader.fetch_seq_len(ref_name));

    CATCH_REQUIRE(ref_end == 10000);

    dorado::secondary::BamFile bam_file(fn_bam, 1);

    CATCH_REQUIRE(bam_file.fp());
    CATCH_REQUIRE(bam_file.idx());
    CATCH_REQUIRE(bam_file.hdr());

    const std::unordered_map<std::string, int32_t> qname2hp{};
    const std::vector<std::string> dtypes{};
    const std::string tag_name{};
    const std::string read_group{};
    constexpr int64_t NUM_DTYPES = 1;
    constexpr int32_t TAG_VALUE = 0;
    constexpr bool TAG_KEEP_MISSING = false;
    constexpr int32_t MIN_MAPQ = 1;
    constexpr bool ROW_PER_READ = false;
    constexpr bool INCLUDE_DWELLS = true;
    constexpr bool INCLUDE_HAPLOTYPE_COLUMN = true;
    constexpr bool INCLUDE_SNP_QV = true;
    constexpr int32_t MAX_READS = 100;
    constexpr bool RIGHT_ALIGN_INSERTIONS = true;
    constexpr double MIN_SNP_ACCURACY = 0.0;

    const dorado::secondary::ReadAlignmentData expected =
            dorado::secondary::calculate_read_alignment(
                    bam_file, ref_name, ref_start, ref_end, qname2hp, NUM_DTYPES, dtypes, tag_name,
                    TAG_VALUE, TAG_KEEP_MISSING, read_group, MIN_MAPQ, ROW_PER_READ, INCLUDE_DWELLS,
                    INCLUDE_HAPLOTYPE_COLUMN, INCLUDE_SNP_QV,
                    dorado::secondary::HaplotagSource::BAM_HAP_TAG, MAX_READS,
                    RIGHT_ALIGN_INSERTIONS, MIN_SNP_ACCURACY);

    const kadayashi::MedakaFeatureMatrixOptions options{
            .include_dwells = INCLUDE_DWELLS,
            .include_haplotype_column = INCLUDE_HAPLOTYPE_COLUMN,
            .include_snp_qv = INCLUDE_SNP_QV,
            .min_mapq = MIN_MAPQ,
            .num_dtypes = NUM_DTYPES,
            .dtypes = dtypes,
            .tag_name = tag_name,
            .tag_value = TAG_VALUE,
            .tag_keep_missing = TAG_KEEP_MISSING,
            .readgroup = read_group,
            .disable_read_packing = ROW_PER_READ,
            .hap_source = kadayashi::USE_BAM_HAP_TAG,
            .max_reads = MAX_READS,
            .right_align_insertions = RIGHT_ALIGN_INSERTIONS,
            .min_snp_accuracy = MIN_SNP_ACCURACY};

    const dorado::secondary::ReadAlignmentData result =
            kadayashi::gen_medaka_feature_matrix_wrapper(bam_file, ref_name, ref_start, ref_end,
                                                         qname2hp, options);

    CATCH_REQUIRE(result.featlen == 7);
    CATCH_CHECK(has_non_zero_feature_value(result, 4));  // Dwells
    CATCH_CHECK(has_non_zero_feature_value(result, 5));  // Haplotags
    CATCH_CHECK(has_non_zero_feature_value(result, 6));  // SNP QV

    compare_feature_matrix_logical_data(result, expected);
}

CATCH_TEST_CASE("kadayashi_featmatgen downsampling", TEST_GROUP) {
    const TempDir temp_dir = make_temp_dir("kadayashi_featmatgen_short_read_downsample");
    const std::filesystem::path temp_in_bam_fn = temp_dir.m_path / "in.aln.bam";

    constexpr int32_t NUM_READS_TARGET = kadayashi::DOWNSAMPLE_READCAP;  // target read depth
    constexpr int32_t NUM_READS_REDUNDANT = 50;
    constexpr int32_t NUM_READS = NUM_READS_TARGET + NUM_READS_REDUNDANT;  // add some extra depth
    const std::string refseq(50, 'A');
    const std::string read_seq(50, 'A');
    const std::vector<std::pair<std::string, std::string>> targets{{"ref", refseq}};

    std::vector<std::string> qnames;
    std::vector<SyntheticBamRecord> records;
    qnames.reserve(NUM_READS);
    records.reserve(NUM_READS);

    for (int32_t i = 0; i < NUM_READS; ++i) {
        qnames.emplace_back("read_" + std::to_string(i));
        records.push_back({
                .qname = qnames.back(),
                .pos = 0,
                .flag = static_cast<uint16_t>((i % 2) == 0 ? 0 : BAM_FREVERSE),
                .mapq = 60,
                .cigar = "50M",
                .seq = read_seq,
                .md = "50",
                .nm = 0,
        });
    }

    write_synthetic_bam(temp_in_bam_fn, targets, records);

    dorado::secondary::BamFile bam_file(temp_in_bam_fn, 1);
    CATCH_REQUIRE(bam_file.fp());
    CATCH_REQUIRE(bam_file.idx());
    CATCH_REQUIRE(bam_file.hdr());

    const std::unordered_map<std::string, int32_t> qname2hp{};
    constexpr uint32_t REF_START = 0;
    constexpr uint32_t REF_END = 50;
    constexpr int64_t NUM_DTYPES = 1;
    constexpr int32_t TAG_VALUE = 0;
    constexpr bool TAG_KEEP_MISSING = false;
    constexpr int32_t MIN_MAPQ = 1;
    constexpr bool ROW_PER_READ = false;
    constexpr bool INCLUDE_DWELLS = false;
    constexpr bool INCLUDE_HAPLOTYPE_COLUMN = false;
    constexpr bool INCLUDE_SNP_QV = false;
    constexpr int32_t MAX_READS = 100;
    constexpr bool RIGHT_ALIGN_INSERTIONS = true;
    constexpr double MIN_SNP_ACCURACY = 0.0;

    const dorado::secondary::ReadAlignmentData expected =
            dorado::secondary::calculate_read_alignment(
                    bam_file, "ref", REF_START, REF_END, qname2hp, NUM_DTYPES, EMPTY_DTYPES,
                    EMPTY_STRING, TAG_VALUE, TAG_KEEP_MISSING, EMPTY_STRING, MIN_MAPQ, ROW_PER_READ,
                    INCLUDE_DWELLS, INCLUDE_HAPLOTYPE_COLUMN, INCLUDE_SNP_QV,
                    dorado::secondary::HaplotagSource::UNPHASED, MAX_READS, RIGHT_ALIGN_INSERTIONS,
                    MIN_SNP_ACCURACY);

    const kadayashi::MedakaFeatureMatrixOptions options = make_medaka_feature_matrix_options(
            INCLUDE_DWELLS, MIN_SNP_ACCURACY, MAX_READS, ROW_PER_READ);
    const dorado::secondary::ReadAlignmentData result =
            kadayashi::gen_medaka_feature_matrix_wrapper(bam_file, "ref", REF_START, REF_END,
                                                         qname2hp, options);

    CATCH_REQUIRE(expected.n_pos == static_cast<int32_t>(std::size(refseq)));
    CATCH_REQUIRE(expected.n_reads <= NUM_READS_TARGET);
    CATCH_REQUIRE(expected.n_reads <= MAX_READS);
    CATCH_CHECK(result.n_pos == expected.n_pos);
    CATCH_CHECK(result.n_reads == expected.n_reads);
    CATCH_CHECK(result.read_ids_left.size() <= static_cast<size_t>(NUM_READS_TARGET));
    CATCH_CHECK(result.read_ids_left.size() <= static_cast<size_t>(MAX_READS));
    CATCH_CHECK(result.read_ids_right.size() <= static_cast<size_t>(NUM_READS_TARGET));
    CATCH_CHECK(result.read_ids_right.size() <= static_cast<size_t>(MAX_READS));
}

CATCH_TEST_CASE("kadayashi_featmatgen no input", TEST_GROUP) {
    const std::filesystem::path test_data_dir =
            get_data_dir("variant") / "test-04-kadayashi-featmatgen";
    const std::filesystem::path fn_in_empty = test_data_dir / "in_aln_almostempty.bam";
    dorado::secondary::BamFile bam_file(fn_in_empty, 1);
    kadayashi::MedakaFeatureMatrixOptions medaka_feature_matrix_options{
            .include_dwells = true,
            .include_haplotype_column = true,
            .include_snp_qv = true,
            .min_mapq = 1,
            .num_dtypes = 1,
            .dtypes = {},
            .tag_name = "",
            .tag_value = 0,
            .tag_keep_missing = false,
            .readgroup = "",
            .disable_read_packing = false,
            .hap_source = kadayashi::USE_TAG_FROM_HASHTABLE,
            .max_reads = 100,
            .right_align_insertions = true,
            .min_snp_accuracy = 0.0};
    const dorado::secondary::ReadAlignmentData result =
            kadayashi::gen_medaka_feature_matrix_wrapper(bam_file, "ref", 0, 1000, {},
                                                         medaka_feature_matrix_options);
    CATCH_CHECK(result.n_pos == 0);
    CATCH_CHECK(result.n_reads == 0);
}

CATCH_TEST_CASE("kadayashi_featmatgen uses SNP accuracy for read filtering", TEST_GROUP) {
    /*
        ref:           0         10        20        30
        perfect:       [=========)
        low_snp_acc:   [=========)  filtered at min_snp_accuracy 0.85
        one_snp:                           [=========)

        Without SNP-accuracy filtering, low_snp_acc overlaps perfect and needs a second lane.
    */
    const std::vector<SyntheticBamRecord> records{
            {.qname = "perfect",
             .pos = 0,
             .mapq = 41,
             .cigar = "10M",
             .seq = "ACGTACGTAA",
             .dwell_tag = make_simple_move_table(10),
             .md = "10",
             .nm = 0},
            {.qname = "low_snp_acc",
             .pos = 0,
             .mapq = 42,
             .cigar = "10M",
             .seq = "ATGTTCGTCA",
             .dwell_tag = make_simple_move_table(10),
             .md = "1C2A3A1",
             .nm = 3},
            {.qname = "one_snp",
             .pos = 20,
             .mapq = 43,
             .cigar = "10M",
             .seq = "ACGTTCGTAA",
             .dwell_tag = make_simple_move_table(10),
             .md = "4A5",
             .nm = 1},
    };

    CATCH_SECTION("filters reads below the SNP accuracy threshold") {
        const kadayashi::MedakaFeatureMatrixOptions options =
                make_medaka_feature_matrix_options(false, 0.85, 100, false);

        const dorado::secondary::ReadAlignmentData result =
                run_feature_matrix_test(records, "ACGTACGTAAACGTACGTAAACGTACGTAA", 0, 30, options);

        CATCH_REQUIRE(result.n_pos == 30);
        CATCH_REQUIRE(result.n_reads == 1);
        CATCH_REQUIRE(result.read_ids_left.size() == 1);
        CATCH_REQUIRE(result.read_ids_right.size() == 1);
        CATCH_CHECK(result.read_ids_left[0] == "perfect");
        CATCH_CHECK(result.read_ids_right[0] == "one_snp");
        CATCH_CHECK(get_feature_matrix_value(result, 0, 0, 0) == 1);
        CATCH_CHECK(get_feature_matrix_value(result, 24, 0, 0) == 4);
    }

    CATCH_SECTION("keeps the low-accuracy read when filtering is disabled") {
        const kadayashi::MedakaFeatureMatrixOptions options =
                make_medaka_feature_matrix_options(false, 0.0, 100, false);

        const dorado::secondary::ReadAlignmentData result =
                run_feature_matrix_test(records, "ACGTACGTAAACGTACGTAAACGTACGTAA", 0, 30, options);

        CATCH_REQUIRE(result.n_pos == 30);
        CATCH_REQUIRE(result.n_reads == 2);
        CATCH_REQUIRE(result.read_ids_left.size() == 2);
        CATCH_REQUIRE(result.read_ids_right.size() == 2);
        CATCH_CHECK(result.read_ids_left[0] == "perfect");
        CATCH_CHECK(result.read_ids_left[1] == "low_snp_acc");
        CATCH_CHECK(result.read_ids_right[0] == "one_snp");
        CATCH_CHECK(result.read_ids_right[1] == "__blank_1");
        CATCH_CHECK(get_feature_matrix_value(result, 1, 1, 0) == 4);
    }
}

CATCH_TEST_CASE("kadayashi_featmatgen skips duplicate read names", TEST_GROUP) {
    /*
        ref:    0         10
        dup #1: [=========)
        dup #2: [=========)  same qname, skipped
    */
    const std::vector<SyntheticBamRecord> records{
            {.qname = "dup",
             .pos = 0,
             .mapq = 41,
             .cigar = "10M",
             .seq = "ACGTACGTAA",
             .dwell_tag = make_simple_move_table(10),
             .md = "10",
             .nm = 0},
            {.qname = "dup",
             .pos = 0,
             .mapq = 42,
             .cigar = "10M",
             .seq = "ACGTACGTAA",
             .dwell_tag = make_simple_move_table(10),
             .md = "10",
             .nm = 0},
    };
    const kadayashi::MedakaFeatureMatrixOptions options =
            make_medaka_feature_matrix_options(false, 0.0, 100, false);

    const dorado::secondary::ReadAlignmentData result =
            run_feature_matrix_test(records, "ACGTACGTAA", 0, 10, options);

    CATCH_REQUIRE(result.n_pos == 10);
    CATCH_CHECK(result.n_reads == 1);
    CATCH_REQUIRE(result.read_ids_left.size() == 1);
    CATCH_REQUIRE(result.read_ids_right.size() == 1);
    CATCH_CHECK(result.read_ids_left[0] == "dup");
    CATCH_CHECK(result.read_ids_right[0] == "dup");
}

CATCH_TEST_CASE("kadayashi_featmatgen reuses lanes at the sentinel boundary", TEST_GROUP) {
    CATCH_SECTION("last and first base overlap, two lanes are produced") {
        /*
            ref:   0    4    9
            left:  [====)
            right:     [====)
        */
        const std::vector<SyntheticBamRecord> records{
                {.qname = "left",
                 .pos = 0,
                 .mapq = 41,
                 .cigar = "5M",
                 .seq = "AAAAA",
                 .md = "5",
                 .nm = 0},
                {.qname = "right",
                 .pos = 4,
                 .mapq = 42,
                 .cigar = "5M",
                 .seq = "CCCCC",
                 .md = "5",
                 .nm = 0},
        };
        const kadayashi::MedakaFeatureMatrixOptions options =
                make_medaka_feature_matrix_options(false, 0.0, 100, false);

        const dorado::secondary::ReadAlignmentData result =
                run_feature_matrix_test(records, "AAAACCCCC", 0, 9, options);

        CATCH_REQUIRE(result.n_pos == 9);
        CATCH_CHECK(result.n_reads == 2);
    }

    CATCH_SECTION("one base gap is still inside the 5bp sentinel and should create a new lane") {
        /*
            ref:   0    5 6    11
            left:  [====)
            gap:        .
            right:       [====)
        */
        const std::vector<SyntheticBamRecord> records{
                {.qname = "left",
                 .pos = 0,
                 .mapq = 41,
                 .cigar = "5M",
                 .seq = "AAAAA",
                 .md = "5",
                 .nm = 0},
                {.qname = "right",
                 .pos = 6,
                 .mapq = 42,
                 .cigar = "5M",
                 .seq = "CCCCC",
                 .md = "5",
                 .nm = 0},
        };
        const kadayashi::MedakaFeatureMatrixOptions options =
                make_medaka_feature_matrix_options(false, 0.0, 100, false);

        const dorado::secondary::ReadAlignmentData result =
                run_feature_matrix_test(records, "AAAAAACCCCC", 0, 11, options);

        CATCH_REQUIRE(result.n_pos == 11);
        CATCH_CHECK(result.n_reads == 2);
    }

    CATCH_SECTION("exact sentinel boundary reuses the lane (two reads are 5bp apart)") {
        /*
            ref:   0    5     10   15
            left:  [====)
            gap:        .....
            right:            [====)
        */
        const std::vector<SyntheticBamRecord> records{
                {.qname = "left",
                 .pos = 0,
                 .mapq = 41,
                 .cigar = "5M",
                 .seq = "AAAAA",
                 .md = "5",
                 .nm = 0},
                {.qname = "right",
                 .pos = 10,
                 .mapq = 42,
                 .cigar = "5M",
                 .seq = "CCCCC",
                 .md = "5",
                 .nm = 0},
        };
        const kadayashi::MedakaFeatureMatrixOptions options =
                make_medaka_feature_matrix_options(false, 0.0, 100, false);

        const dorado::secondary::ReadAlignmentData result =
                run_feature_matrix_test(records, "AAAAAAAAAACCCCC", 0, 15, options);

        CATCH_REQUIRE(result.n_pos == 15);
        CATCH_CHECK(result.n_reads == 1);
        CATCH_REQUIRE(result.read_ids_left.size() == 1);
        CATCH_REQUIRE(result.read_ids_right.size() == 1);
        CATCH_CHECK(result.read_ids_left[0] == "left");
        CATCH_CHECK(result.read_ids_right[0] == "right");
        CATCH_CHECK(get_feature_matrix_value(result, 0, 0, 0) == 1);
        CATCH_CHECK(get_feature_matrix_value(result, 10, 0, 0) == 2);
    }
}

CATCH_TEST_CASE("kadayashi_featmatgen caps lanes while packing", TEST_GROUP) {
    /*
        ref: 0 1 2       12
        r1:  [=========)
        r2:   [=========)
        r3:    [=========)  over max_reads=2
    */
    const std::vector<SyntheticBamRecord> records{
            {.qname = "r1",
             .pos = 0,
             .mapq = 41,
             .cigar = "10M",
             .seq = "AAAAAAAAAA",
             .md = "10",
             .nm = 0},
            {.qname = "r2",
             .pos = 1,
             .mapq = 42,
             .cigar = "10M",
             .seq = "CCCCCCCCCC",
             .md = "10",
             .nm = 0},
            {.qname = "r3",
             .pos = 2,
             .mapq = 43,
             .cigar = "10M",
             .seq = "GGGGGGGGGG",
             .md = "10",
             .nm = 0},
    };
    const kadayashi::MedakaFeatureMatrixOptions options =
            make_medaka_feature_matrix_options(false, 0.0, 2, false);

    const dorado::secondary::ReadAlignmentData result =
            run_feature_matrix_test(records, "AAAAAAAAAAAA", 0, 12, options);

    CATCH_REQUIRE(result.n_pos == 12);
    CATCH_CHECK(result.n_reads == 2);
    CATCH_CHECK(result.read_ids_left.size() == 2);
    CATCH_CHECK(result.read_ids_right.size() == 2);
}

CATCH_TEST_CASE("kadayashi_featmatgen preserves over-cap read suffixes", TEST_GROUP) {
    /*
        ref: 0    5    10   15        25
        r1:  [=========)
        r2:       [===================)  max_reads=1, suffix starts after sentinel
        lane: [r1      ).....[r2 suffix)
    */
    const std::vector<SyntheticBamRecord> records{
            {.qname = "r1",
             .pos = 0,
             .mapq = 41,
             .cigar = "10M",
             .seq = "AAAAAAAAAA",
             .md = "10",
             .nm = 0},
            {.qname = "r2",
             .pos = 5,
             .mapq = 42,
             .cigar = "20M",
             .seq = "CCCCCCCCCCCCCCCCCCCC",
             .md = "20",
             .nm = 0},
    };
    const kadayashi::MedakaFeatureMatrixOptions options =
            make_medaka_feature_matrix_options(false, 0.0, 1, false);

    const dorado::secondary::ReadAlignmentData result =
            run_feature_matrix_test(records, "AAAAAAAAAACCCCCCCCCCCCCCC", 0, 25, options);

    CATCH_REQUIRE(result.n_pos == 25);
    CATCH_REQUIRE(result.n_reads == 1);
    CATCH_REQUIRE(result.read_ids_left.size() == 1);
    CATCH_REQUIRE(result.read_ids_right.size() == 1);
    CATCH_CHECK(result.read_ids_left[0] == "r1");
    CATCH_CHECK(result.read_ids_right[0] == "r2");
    CATCH_CHECK(get_feature_matrix_value(result, 14, 0, 0) == 0);
    CATCH_CHECK(get_feature_matrix_value(result, 15, 0, 0) == 2);
    CATCH_CHECK(get_feature_matrix_value(result, 24, 0, 0) == 2);
}

CATCH_TEST_CASE("kadayashi_featmatgen wrapper handles empty outputs", TEST_GROUP) {
    CATCH_SECTION("no records in the queried interval") {
        /*
            query: [0       10)
            read:                      [20      30)
        */
        const std::vector<SyntheticBamRecord> records{
                {.qname = "outside",
                 .pos = 20,
                 .mapq = 41,
                 .cigar = "10M",
                 .seq = "ACGTACGTAA",
                 .dwell_tag = make_simple_move_table(10),
                 .md = "10",
                 .nm = 0},
        };
        const kadayashi::MedakaFeatureMatrixOptions options =
                make_medaka_feature_matrix_options(false, 0.0, 100, false);

        const dorado::secondary::ReadAlignmentData result =
                run_feature_matrix_test(records, "ACGTACGTAAACGTACGTAAACGTACGTAA", 0, 10, options);

        CATCH_CHECK(result.n_pos == 0);
        CATCH_CHECK(result.n_reads == 0);
    }

    CATCH_SECTION("all input records are filtered out") {
        /*
            ref:      0         10
            low_mapq: [=========)  filtered because mapq < min_mapq
        */
        const std::vector<SyntheticBamRecord> records{
                {.qname = "low_mapq",
                 .pos = 0,
                 .mapq = 0,
                 .cigar = "10M",
                 .seq = "ACGTACGTAA",
                 .dwell_tag = make_simple_move_table(10),
                 .md = "10",
                 .nm = 0},
        };
        const kadayashi::MedakaFeatureMatrixOptions options =
                make_medaka_feature_matrix_options(false, 0.0, 100, false);

        const dorado::secondary::ReadAlignmentData result =
                run_feature_matrix_test(records, "ACGTACGTAA", 0, 10, options);

        CATCH_CHECK(result.n_pos == 0);
        CATCH_CHECK(result.n_reads == 0);
    }
}

}  // namespace kadayashi::tests
