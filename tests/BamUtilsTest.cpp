#include "TestUtils.h"
#include "utils/bam_utils.h"

#include <catch2/catch.hpp>

#include <filesystem>

#define TEST_GROUP "[bam_utils]"

namespace fs = std::filesystem;

TEST_CASE("BamUtilsTest: fetch keys from PG header", TEST_GROUP) {
    fs::path aligner_test_dir = fs::path(get_data_dir("aligner_test"));
    auto sam = aligner_test_dir / "basecall.sam";

    auto keys = dorado::utils::extract_pg_keys_from_hdr(sam.string(), {"PN", "CL", "VN"});
    CHECK(keys["PN"] == "dorado");
    CHECK(keys["VN"] == "0.2.3+0f041c4+dirty");
    CHECK(keys["CL"] ==
          "dorado basecaller dna_r9.4.1_e8_hac@v3.3 ./tests/data/pod5 -x cpu --modified-bases "
          "5mCG");
}
