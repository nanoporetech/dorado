#include "TestUtils.h"
#include "utils/overlap.h"
#include "utils/paf_utils.h"

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#define TEST_GROUP "[paf_utils]"

CATCH_TEST_CASE("PafUtilsTest: Test round trip PAF loading and writing", TEST_GROUP) {
    const std::filesystem::path paf_utils_test_dir = get_data_dir("paf_utils");
    const std::filesystem::path paf = paf_utils_test_dir / "test.paf";

    std::ifstream file(paf.string());
    CATCH_REQUIRE(file.is_open());

    std::string line;
    while (std::getline(file, line)) {
        const dorado::utils::PafEntry paf_entry = dorado::utils::parse_paf(line);
        const std::string serialized_paf = dorado::utils::serialize_paf(paf_entry);
        CATCH_CHECK(line == serialized_paf);
    }
}

CATCH_TEST_CASE("PafUtilsTest: Test aux loading", TEST_GROUP) {
    const std::filesystem::path paf_utils_test_dir = get_data_dir("paf_utils");
    const std::filesystem::path paf = paf_utils_test_dir / "test.paf";

    std::ifstream file(paf.string());
    CATCH_REQUIRE(file.is_open());

    std::string line;
    while (std::getline(file, line)) {
        const dorado::utils::PafEntry paf_entry = dorado::utils::parse_paf(line);
        const std::string_view cg = dorado::utils::paf_aux_get(paf_entry, "cg", 'Z');
        CATCH_CHECK_FALSE(cg.empty());
        const std::vector<dorado::CigarOp> ops = dorado::parse_cigar_from_string(cg);
        CATCH_CHECK_FALSE(ops.empty());
    }
}

CATCH_TEST_CASE("PafUtilsTest: Test serialize_to_paf", TEST_GROUP) {
    CATCH_SECTION("Record with forward strand mapping.") {
        dorado::utils::Overlap ovl;
        ovl.qstart = 0;
        ovl.qend = 100;
        ovl.qlen = 200;
        ovl.tstart = 300;
        ovl.tend = 400;
        ovl.tlen = 500;
        ovl.fwd = true;

        const std::vector<dorado::CigarOp> cigar = {
                {dorado::CigarOpType::EQ, 50},
                {dorado::CigarOpType::D, 1},
                {dorado::CigarOpType::EQ, 50},
        };

        std::ostringstream oss;
        dorado::utils::serialize_to_paf(oss, "query01", "target02", ovl, 1, 2, 3, cigar);

        const std::string expected =
                "query01\t200\t0\t100\t+\ttarget02\t500\t300\t400\t1\t2\t3\tcg:Z:50=1D50=";

        CATCH_CHECK(expected == oss.str());
    }

    CATCH_SECTION("Reverse complement record.") {
        dorado::utils::Overlap ovl;
        ovl.qstart = 0;
        ovl.qend = 100;
        ovl.qlen = 200;
        ovl.tstart = 300;
        ovl.tend = 400;
        ovl.tlen = 500;
        ovl.fwd = false;

        const std::vector<dorado::CigarOp> cigar = {
                {dorado::CigarOpType::EQ, 50},
                {dorado::CigarOpType::I, 1},
                {dorado::CigarOpType::X, 49},
        };

        std::ostringstream oss;
        dorado::utils::serialize_to_paf(oss, "query3", "target4", ovl, 1, 2, 3, cigar);

        const std::string expected =
                "query3\t200\t0\t100\t-\ttarget4\t500\t300\t400\t1\t2\t3\tcg:Z:50=1I49X";

        CATCH_CHECK(expected == oss.str());
    }
}
