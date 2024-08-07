#include "TestUtils.h"
#include "utils/alignment_utils.h"
#include "utils/paf_utils.h"

#include <catch2/catch.hpp>
#include <htslib/sam.h>

#include <filesystem>
#include <fstream>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#define TEST_GROUP "[paf_utils]"

namespace fs = std::filesystem;
using namespace dorado;

TEST_CASE("PafUtilsTest: Test PAF loading", TEST_GROUP) {
    fs::path paf_utils_test_dir = fs::path(get_data_dir("paf_utils"));
    auto paf = paf_utils_test_dir / "test.paf";

    std::ifstream file(paf.string());
    REQUIRE(file.is_open());

    std::string line;
    while (std::getline(file, line)) {
        auto paf_entry = utils::parse_paf(line);
        auto serialized_paf = utils::serialize_paf(paf_entry);
        CHECK(line == serialized_paf);
    }
}

TEST_CASE("PafUtilsTest: Test aux loading", TEST_GROUP) {
    fs::path paf_utils_test_dir = fs::path(get_data_dir("paf_utils"));
    auto paf = paf_utils_test_dir / "test.paf";

    std::ifstream file(paf.string());
    REQUIRE(file.is_open());

    std::string line;
    while (std::getline(file, line)) {
        auto paf_entry = utils::parse_paf(line);
        auto cg_ptr = utils::paf_aux_get(paf_entry, "cg", 'Z');
        CHECK(!cg_ptr.empty());
        auto ops = parse_cigar_from_string(cg_ptr);
        CHECK(!ops.empty());
    }
}
