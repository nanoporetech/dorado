#include "TestUtils.h"
#include "blocked_bloom_filter.h"
#include "hts_utils/FastxRandomReader.h"
#include "local_haplotagging.h"
#include "secondary/common/bam_file.h"
#include "sequence_utility.h"
#include "types.h"

#include <catch2/catch_test_macros.hpp>
#include <htslib/faidx.h>
#include <htslib/khash.h>
#include <htslib/khash_str2int.h>
#include <htslib/sam.h>
#include <stdint.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <unordered_map>

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

}  // namespace

CATCH_TEST_CASE("kadayashi blocked bloom filter basic operation", TEST_GROUP) {
    kadayashi::BlockedBloomFilter bf(4, 16);  // tiny
    bf.enable();
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

CATCH_TEST_CASE("kadayashi dvr and simple, normal case", TEST_GROUP) {
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
    dorado::secondary::BamFile bam_reader(fn_bam);
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

    CATCH_SECTION("kadayashi_dvr_single_region_wrapper") {
        const std::unordered_map<std::string, int32_t> result =
                kadayashi::kadayashi_dvr_single_region_wrapper(
                        bam_reader.fp(), bam_reader.idx(), bam_reader.hdr(),
                        fastx_reader.get_raw_faidx_ptr(), "chr20", 0, 9999,
                        DISABLE_INTERVAL_EXPANSION, MIN_BASE_QUALITY, MIN_VARCALL_COVERAGE,
                        MIN_VARCALL_FRACTION, MAX_CLIPPING, MIN_STRAND_COV, MIN_STRAND_COV_FRAC,
                        MAX_GAPCOMPRESSED_SEQDIV);
        CATCH_CHECK(compare_haptags(result, expected));
    }

    CATCH_SECTION("kadayashi_simple_single_region_wrapper") {
        const std::unordered_map<std::string, int32_t> result =
                kadayashi::kadayashi_simple_single_region_wrapper(
                        bam_reader.fp(), bam_reader.idx(), bam_reader.hdr(),
                        fastx_reader.get_raw_faidx_ptr(), "chr20", 0, 9999,
                        DISABLE_INTERVAL_EXPANSION, MIN_BASE_QUALITY, MIN_VARCALL_COVERAGE,
                        MIN_VARCALL_FRACTION, MAX_CLIPPING, MIN_STRAND_COV, MIN_STRAND_COV_FRAC,
                        MAX_GAPCOMPRESSED_SEQDIV);
        CATCH_CHECK(compare_haptags(result, expected));
    }
}

CATCH_TEST_CASE("kadayashi dvr and simple, empty region", TEST_GROUP) {
    // Input data.
    const std::filesystem::path test_data_dir = get_data_dir("variant") / "test-02-supertiny";
    const std::filesystem::path fn_bam = test_data_dir / "in.aln.bam";
    const std::filesystem::path fn_ref = test_data_dir / "in.ref.fasta.gz";

    const std::unordered_map<std::string, int32_t> expected{};

    // Open the input files.
    dorado::secondary::BamFile bam_reader(fn_bam);
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

    CATCH_SECTION("kadayashi_dvr_single_region_wrapper empty region") {
        // UUT.
        const std::unordered_map<std::string, int32_t> result =
                kadayashi::kadayashi_dvr_single_region_wrapper(
                        bam_reader.fp(), bam_reader.idx(), bam_reader.hdr(),
                        fastx_reader.get_raw_faidx_ptr(), "chr20", 200000, 200001,
                        DISABLE_INTERVAL_EXPANSION, MIN_BASE_QUALITY, MIN_VARCALL_COVERAGE,
                        MIN_VARCALL_FRACTION, MAX_CLIPPING, MIN_STRAND_COV, MIN_STRAND_COV_FRAC,
                        MAX_GAPCOMPRESSED_SEQDIV);
        CATCH_CHECK(compare_haptags(result, expected));
    }

    CATCH_SECTION("kadayashi_dvr_single_region_wrapper bad coordinate span") {
        // UUT.
        const std::unordered_map<std::string, int32_t> result =
                kadayashi::kadayashi_dvr_single_region_wrapper(
                        bam_reader.fp(), bam_reader.idx(), bam_reader.hdr(),
                        fastx_reader.get_raw_faidx_ptr(), "chr20", 200000, 199999,
                        DISABLE_INTERVAL_EXPANSION, MIN_BASE_QUALITY, MIN_VARCALL_COVERAGE,
                        MIN_VARCALL_FRACTION, MAX_CLIPPING, MIN_STRAND_COV, MIN_STRAND_COV_FRAC,
                        MAX_GAPCOMPRESSED_SEQDIV);
        CATCH_CHECK(compare_haptags(result, expected));
    }

    CATCH_SECTION("kadayashi_simple_single_region_wrapper empty region") {
        // UUT.
        const std::unordered_map<std::string, int32_t> result =
                kadayashi::kadayashi_simple_single_region_wrapper(
                        bam_reader.fp(), bam_reader.idx(), bam_reader.hdr(),
                        fastx_reader.get_raw_faidx_ptr(), "chr20", 200000, 200001,
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
                        fastx_reader.get_raw_faidx_ptr(), "chr20", 200000, 199999,
                        DISABLE_INTERVAL_EXPANSION, MIN_BASE_QUALITY, MIN_VARCALL_COVERAGE,
                        MIN_VARCALL_FRACTION, MAX_CLIPPING, MIN_STRAND_COV, MIN_STRAND_COV_FRAC,
                        MAX_GAPCOMPRESSED_SEQDIV);
        CATCH_CHECK(compare_haptags(result, expected));
    }
}

CATCH_TEST_CASE("kadayashi dvr and simple nonexistent chromosome", TEST_GROUP) {
    // Input data.
    const std::filesystem::path test_data_dir = get_data_dir("variant") / "test-02-supertiny";
    const std::filesystem::path fn_bam = test_data_dir / "in.aln.bam";
    const std::filesystem::path fn_ref = test_data_dir / "in.ref.fasta.gz";

    const std::unordered_map<std::string, int32_t> expected{};

    // Open the input files.
    dorado::secondary::BamFile bam_reader(fn_bam);
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

    CATCH_SECTION("kadayashi_dvr_single_region_wrapper empty region") {
        const std::unordered_map<std::string, int32_t> result =
                kadayashi::kadayashi_dvr_single_region_wrapper(
                        bam_reader.fp(), bam_reader.idx(), bam_reader.hdr(),
                        fastx_reader.get_raw_faidx_ptr(), "Nonexistent", 200000, 200001,
                        DISABLE_INTERVAL_EXPANSION, MIN_BASE_QUALITY, MIN_VARCALL_COVERAGE,
                        MIN_VARCALL_FRACTION, MAX_CLIPPING, MIN_STRAND_COV, MIN_STRAND_COV_FRAC,
                        MAX_GAPCOMPRESSED_SEQDIV);
        CATCH_CHECK(compare_haptags(result, expected));
    }

    CATCH_SECTION("kadayashi_simple_single_region_wrapper empty region") {
        const std::unordered_map<std::string, int32_t> result =
                kadayashi::kadayashi_simple_single_region_wrapper(
                        bam_reader.fp(), bam_reader.idx(), bam_reader.hdr(),
                        fastx_reader.get_raw_faidx_ptr(), "Nonexistent", 200000, 200001,
                        DISABLE_INTERVAL_EXPANSION, MIN_BASE_QUALITY, MIN_VARCALL_COVERAGE,
                        MIN_VARCALL_FRACTION, MAX_CLIPPING, MIN_STRAND_COV, MIN_STRAND_COV_FRAC,
                        MAX_GAPCOMPRESSED_SEQDIV);
        CATCH_CHECK(compare_haptags(result, expected));
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
            .variants = {
                    // 0-index
                    {true, true, 93, 60, "C", {"T"}, {'1', '0'}},
                    {true, true, 305, 60, "G", {"A"}, {'1', '0'}},
                    {false, false, 775, 0, "TC", {"T"}, {'0', '1'}},
                    {false, false, 809, 0, "AC", {"A"}, {'0', '1'}},
                    {false, false, 1002, 0, "AC", {"A"}, {'0', '1'}},
                    {true, true, 1471, 60, "T", {"G"}, {'0', '1'}},
                    {false, false, 1613, 0, "TC", {"T"}, {'0', '1'}},
                    {false, false, 1619, 0, "CG", {"C"}, {'0', '1'}},
                    {false, false, 1829, 0, "CTTT", {"C"}, {'0', '1'}},
                    {false, false, 1958, 0, "T", {"G"}, {'0', '1'}},
                    {true, true, 2101, 60, "A", {"G"}, {'0', '1'}},
                    {true, true, 2125, 60, "C", {"G"}, {'0', '1'}},
                    {true, true, 2437, 60, "T", {"C"}, {'0', '1'}},
                    {true, true, 2445, 60, "G", {"A"}, {'0', '1'}},
                    {true, false, 2966, 60, "G", {"C"}, {'1', '1'}},
                    {true, true, 3175, 60, "G", {"T"}, {'0', '1'}},
                    {false, false, 3514, 0, "CT", {"C"}, {'0', '1'}},
                    {true, false, 3734, 60, "G", {"C"}, {'1', '1'}},
                    {false, false, 3961, 0, "A", {"AC"}, {'0', '1'}},
                    {false, false, 4695, 0, "A", {"T"}, {'0', '1'}},
                    //{false, false, 4999, 0, {"A","AAAAT"}, {"AAAAT","A"}, {'2','1'}},
                    {false, false, 4999, 0, "AAAAT", {"AAAATAAAT", "A"}, {'2', '1'}},
                    {false, false, 5125, 0, "CTTT", {"C"}, {'0', '1'}},
                    {true, true, 5386, 60, "C", {"G"}, {'0', '1'}},
                    {false, false, 5985, 0, "CA", {"C"}, {'0', '1'}},
                    {false, false, 6947, 0, "CGTGT", {"C"}, {'0', '1'}},
                    {true, true, 7429, 60, "C", {"T"}, {'0', '1'}},
                    {false, false, 7690, 0, "TCC", {"T"}, {'0', '1'}},
                    {false, false, 7912, 0, "C", {"CT"}, {'0', '1'}},
                    {false, false, 8319, 0, "GAA", {"G"}, {'0', '1'}},
                    {false, false, 8973, 0, "C", {"CG"}, {'0', '1'}},
            }};

    // Open the input files.
    dorado::secondary::BamFile bam_reader(fn_bam);
    dorado::hts_io::FastxRandomReader fastx_reader(fn_ref);

    CATCH_REQUIRE(bam_reader.fp());
    CATCH_REQUIRE(bam_reader.idx());
    CATCH_REQUIRE(bam_reader.hdr());
    CATCH_REQUIRE(fastx_reader.get_raw_faidx_ptr());

    const kadayashi::pileup_pars_t pp{.max_clipping = 100000};

    CATCH_SECTION("simple phasing varcall") {
        const kadayashi::varcall_result_t result = kadayashi::kadayashi_phase_and_varcall_wrapper(
                bam_reader.fp(), bam_reader.idx(), bam_reader.hdr(),
                fastx_reader.get_raw_faidx_ptr(), "chr20", 0, 9999, pp.disable_region_expansion,
                pp.min_base_quality, pp.min_varcall_coverage, pp.min_varcall_fraction,
                pp.max_clipping, 1 /*min strand cov*/, 0.033f, pp.max_gapcompressed_seqdiv, false);
        CATCH_CHECK(compare_haptags(result.qname2hp, expected.qname2hp));
    }

    CATCH_SECTION("dvr phasing") {
        // dvr and simple phasing share the same variant calling step
        const kadayashi::varcall_result_t result = kadayashi::kadayashi_phase_and_varcall_wrapper(
                bam_reader.fp(), bam_reader.idx(), bam_reader.hdr(),
                fastx_reader.get_raw_faidx_ptr(), "chr20", 0, 9999, pp.disable_region_expansion,
                pp.min_base_quality, pp.min_varcall_coverage, pp.min_varcall_fraction,
                pp.max_clipping, 1 /*min strand cov*/, 0.033f, pp.max_gapcompressed_seqdiv, true);
        CATCH_CHECK(compare_haptags(result.qname2hp, expected.qname2hp));
    }

    CATCH_SECTION("use wrong clipping threshold") {
        const kadayashi::varcall_result_t result3 = kadayashi::kadayashi_phase_and_varcall_wrapper(
                bam_reader.fp(), bam_reader.idx(), bam_reader.hdr(),
                fastx_reader.get_raw_faidx_ptr(), "chr20", 0, 9999, pp.disable_region_expansion,
                pp.min_base_quality, pp.min_varcall_coverage, pp.min_varcall_fraction, 100,
                1 /*min strand cov*/, 0.033f, pp.max_gapcompressed_seqdiv, false);
        CATCH_CHECK(result3.variants.empty());
    }
}

}  // namespace kadayashi::tests
