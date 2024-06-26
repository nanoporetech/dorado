#include "alignment/BedFile.h"

#include "TestUtils.h"

#include <catch2/catch.hpp>

#define CUT_TAG "[BedFile]"

TEST_CASE(CUT_TAG ": test bedfile loading", CUT_TAG) {
    auto data_dir = get_data_dir("bedfile_test");
    auto test_file = (data_dir / "test_bed.bed").string();
    dorado::alignment::BedFile bed;
    bed.load(test_file);
    const auto& entries = bed.entries("Lambda");
    REQUIRE(entries.size() == size_t(4));
    std::vector<size_t> expected_starts{40000, 41000, 80000, 81000};
    std::vector<char> expected_dir{'+', '+', '-', '+'};
    size_t expected_length = 1000;
    for (size_t i = 0; i < entries.size(); ++i) {
        REQUIRE(entries[i].start == expected_starts[i]);
        REQUIRE(entries[i].end == expected_starts[i] + expected_length);
        REQUIRE(entries[i].strand == expected_dir[i]);
    }
}
