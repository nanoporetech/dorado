#include "TestUtils.h"
#include "hts_utils/FastxRandomReader.h"
#include "local_haplotagging.h"
#include "secondary/common/bam_file.h"
#include "types.h"

#include <catch2/catch_test_macros.hpp>
#include <htslib/faidx.h>
#include <htslib/khash.h>
#include <htslib/khash_str2int.h>
#include <htslib/sam.h>
#include <stdint.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <unordered_map>

#define TEST_GROUP "[KadayashiInterfaceTest]"

namespace kadayashi::tests {

CATCH_TEST_CASE("kadayashi_dvr_1 normal case", TEST_GROUP) {
    // Input data.
    const std::filesystem::path test_data_dir = get_data_dir("variant") / "test-02-supertiny";
    const std::filesystem::path fn_bam = test_data_dir / "in.aln.bam";
    const std::filesystem::path fn_ref = test_data_dir / "in.ref.fasta.gz";

    const std::unordered_map<std::string, int32_t> expected{
            {"1e70cda3-c41f-4d19-9c14-94d8d64e619c", 2},
            {"61ab09d6-072f-4ab2-b14b-b0a1e38a3419", 2},
            {"563ecca1-30dd-4dd9-991a-d417d827c803", 1},
            {"4fd81aa2-cb77-4994-a8a5-70e6228f255e", 1},
            {"a27cad27-2297-40d4-8666-40a4742eb2ed", 2},
            {"e0af6c87-8655-4603-97b7-0ad5ba860df2", 1},
            {"7d23577c-5c93-4d41-83bd-b652e687deee", 1},
            {"627ea9e1-5204-4a2c-ae54-1e1be8bbbbe6", 2},
            {"ac863a7d-932e-42fa-91c1-7814d7f810f9", 2},
            {"b4139858-e420-4780-94e6-375542c2d2e8", 1},
            {"dbe9785a-fa25-454c-9960-fd65fb99a040", 2},
            {"3fdc1b9b-7186-411e-af92-e93a1086754c", 2},
            {"7b2095d4-08f7-448d-aa9d-55c9568fb49d", 2},
            {"c488f4c5-1639-4be1-92f6-948f29b7d822", 2},
            {"02551418-20c9-4b4b-9d1b-9bee36342895", 2},
            {"de45db56-e704-4524-af88-06a2f98c270e", 2},
            {"49b05d0d-97ac-449e-804b-35b35e05ce28", 1},
            {"e7e27cb5-1144-49dd-8ec4-09a75937a091", 1},
            {"3d7a9813-67be-4b84-b66a-0269aa108340", 2},
            {"d5560893-59c8-417c-a929-d62b4d19a1ca", 2},
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

    // UUT.
    const std::unordered_map<std::string, int32_t> result =
            kadayashi::kadayashi_dvr_single_region_wrapper(
                    bam_reader.fp(), bam_reader.idx(), bam_reader.hdr(),
                    fastx_reader.get_raw_faidx_ptr(), "chr20", 1, 10000, DISABLE_INTERVAL_EXPANSION,
                    MIN_BASE_QUALITY, MIN_VARCALL_COVERAGE, MIN_VARCALL_FRACTION, MAX_CLIPPING,
                    MIN_STRAND_COV, MIN_STRAND_COV_FRAC, MAX_GAPCOMPRESSED_SEQDIV);

    CATCH_CHECK(result == expected);
}

CATCH_TEST_CASE("kadayashi_dvr_1 empty region", TEST_GROUP) {
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

    // UUT.
    const std::unordered_map<std::string, int32_t> result =
            kadayashi::kadayashi_dvr_single_region_wrapper(
                    bam_reader.fp(), bam_reader.idx(), bam_reader.hdr(),
                    fastx_reader.get_raw_faidx_ptr(), "chr20", 200000, 200001,
                    DISABLE_INTERVAL_EXPANSION, MIN_BASE_QUALITY, MIN_VARCALL_COVERAGE,
                    MIN_VARCALL_FRACTION, MAX_CLIPPING, MIN_STRAND_COV, MIN_STRAND_COV_FRAC,
                    MAX_GAPCOMPRESSED_SEQDIV);

    CATCH_CHECK(result == expected);
}

CATCH_TEST_CASE("kadayashi_dvr_1 nonexistent chromosome", TEST_GROUP) {
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

    // UUT.
    const std::unordered_map<std::string, int32_t> result =
            kadayashi::kadayashi_dvr_single_region_wrapper(
                    bam_reader.fp(), bam_reader.idx(), bam_reader.hdr(),
                    fastx_reader.get_raw_faidx_ptr(), "Nonexistent", 200000, 200001,
                    DISABLE_INTERVAL_EXPANSION, MIN_BASE_QUALITY, MIN_VARCALL_COVERAGE,
                    MIN_VARCALL_FRACTION, MAX_CLIPPING, MIN_STRAND_COV, MIN_STRAND_COV_FRAC,
                    MAX_GAPCOMPRESSED_SEQDIV);

    CATCH_CHECK(result == expected);
}

}  // namespace kadayashi::tests
