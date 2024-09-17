#include "alignment/bed_file.h"

#include "TestUtils.h"
#include "utils/stream_utils.h"

#include <catch2/catch.hpp>

#include <sstream>
#include <string>

#define CUT_TAG "[dorado::alignment::BedFile]"

namespace dorado::alignment::bed_file::test {

namespace {
const BedFile::Entry LAMBDA_1{"Lambda\t1234\t2345\tcomment1\t100\t+", 1234, 2345, '+'};
const BedFile::Entry LAMBDA_2{"Lambda\t3456\t4567\tcomment2\t101\t-", 3456, 4567, '-'};
const BedFile::Entry RANDOM_1{"Random\t3456\t4567\tcomment3\t102\t-", 3456, 4567, '-'};
const BedFile::Entry RANDOM_2{"Random\t5678\t6789\tcomment4\t101\t+", 5678, 6789, '+'};
}  // namespace

TEST_CASE(CUT_TAG " load from valid file all entries loaded", CUT_TAG) {
    auto data_dir = get_data_dir("bedfile_test");
    auto test_file = (data_dir / "test_bed.bed").string();
    dorado::alignment::BedFile bed;
    CHECK(bed.load(test_file));
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

TEST_CASE(CUT_TAG " load from stream with valid input does not throw.", CUT_TAG) {
    dorado::alignment::BedFile cut{};
    std::istringstream input_stream{"Lambda\t1234\t2345\tcomment\t100\t+"};

    REQUIRE_NOTHROW(cut.load(input_stream));
}

TEST_CASE(CUT_TAG " load with valid single record creates single entry.", CUT_TAG) {
    dorado::alignment::BedFile cut{};
    std::istringstream input_stream{LAMBDA_1.bed_line};
    CHECK(cut.load(input_stream));

    const auto entries = cut.entries("Lambda");

    REQUIRE(entries.size() == 1);
    CHECK(entries[0] == LAMBDA_1);
}

TEST_CASE(CUT_TAG " load with 'track' header line plus valid single record creates single entry.",
          CUT_TAG) {
    dorado::alignment::BedFile cut{};
    std::istringstream input_stream{"track blah\tblah2\n" + LAMBDA_1.bed_line};
    CHECK(cut.load(input_stream));

    const auto entries = cut.entries("Lambda");

    REQUIRE(entries.size() == 1);
    CHECK(entries[0] == LAMBDA_1);
}

TEST_CASE(CUT_TAG
          " load with 'browser' header line plus valid single record creates single "
          "entry.",
          CUT_TAG) {
    dorado::alignment::BedFile cut{};
    std::istringstream input_stream{"browser blah\tblah2\n" + LAMBDA_1.bed_line};
    CHECK(cut.load(input_stream));

    const auto entries = cut.entries("Lambda");

    REQUIRE(entries.size() == 1);
    CHECK(entries[0] == LAMBDA_1);
}

TEST_CASE(CUT_TAG
          " load with '#' comment header line plus valid single record creates single "
          "entry.",
          CUT_TAG) {
    dorado::alignment::BedFile cut{};
    std::istringstream input_stream{"# col1, col2, col3\n" + LAMBDA_1.bed_line};
    CHECK(cut.load(input_stream));

    const auto entries = cut.entries("Lambda");

    REQUIRE(entries.size() == 1);
    CHECK(entries[0] == LAMBDA_1);
}

TEST_CASE(CUT_TAG
          " load with multiple header lines plus valid single record creates single "
          "entry.",
          CUT_TAG) {
    dorado::alignment::BedFile cut{};
    std::istringstream input_stream{"browser blah\tblah2\ntrack blah\tblah2\n# col1, col2, col3\n" +
                                    LAMBDA_1.bed_line};
    CHECK(cut.load(input_stream));

    const auto entries = cut.entries("Lambda");

    REQUIRE(entries.size() == 1);
    CHECK(entries[0] == LAMBDA_1);
}

TEST_CASE(CUT_TAG " load with header line after bed data line returns false.", CUT_TAG) {
    dorado::alignment::BedFile cut{};
    std::istringstream input_stream{LAMBDA_1.bed_line + "\nbrowser blah"};

    utils::SuppressStdout suppress_error_message{};
    REQUIRE_FALSE(cut.load(input_stream));
}

TEST_CASE(CUT_TAG " load with two records same genome creates two entries for genome.", CUT_TAG) {
    dorado::alignment::BedFile cut{};
    std::istringstream input_stream{LAMBDA_1.bed_line + "\n" + LAMBDA_2.bed_line};
    CHECK(cut.load(input_stream));

    const auto entries = cut.entries("Lambda");

    REQUIRE(entries.size() == 2);
    CHECK(entries[0] == LAMBDA_1);
    CHECK(entries[1] == LAMBDA_2);
}

TEST_CASE(CUT_TAG
          " load with 4 records, 2 genomes with 2 entries each, returns the correct "
          "entries for genomes.",
          CUT_TAG) {
    dorado::alignment::BedFile cut{};
    std::istringstream input_stream{LAMBDA_1.bed_line + "\n" + RANDOM_1.bed_line + "\n" +
                                    LAMBDA_2.bed_line + "\n" + RANDOM_2.bed_line};
    CHECK(cut.load(input_stream));

    auto entries = cut.entries("Lambda");
    REQUIRE(entries.size() == 2);
    CHECK(entries[0] == LAMBDA_1);
    CHECK(entries[1] == LAMBDA_2);

    entries = cut.entries("Random");
    REQUIRE(entries.size() == 2);
    CHECK(entries[0] == RANDOM_1);
    CHECK(entries[1] == RANDOM_2);
}

TEST_CASE(CUT_TAG " load from stream. Parameterised testing.", CUT_TAG) {
    // clang-format off
    auto [line, genome, start, end, strand, valid] = GENERATE(
        table<std::string, std::string, std::size_t, std::size_t, char, bool>({
            {"Lambda\t1234\t2345\tcomment1\t100\t+", "Lambda", 1234, 2345, '+', true},
            {"Lambda\t1234\t2345", "Lambda", 1234, 2345, '.', true},
            {"Lambda 1234 2345", "Lambda", 0, 0, 0, false},
            {"Lambda\tabc\t2345", "Lambda", 0, 0, 0, false},
            {"Lambda\t1234\t2345\tcomment with spaces", "Lambda", 1234, 2345, '.', true},
            {"Lambda\t12345\t23456\tspaces column\t100", "Lambda", 12345, 23456, '.', true},
            {"Lambda\t1234\t2345\tinvalid strand\t100\tTTT", "Lambda", 0, 0, 0, false},
            {"12Fields\t1\t2\tab c\t100\t+\t0\t0\t1,2,3\t0\t123,234\t456,567", "12Fields", 1, 2, '+', true},
            {"13Fields\t1\t2\tab c\t100\t+\t0\t0\t1,2,3\t0\t1,2\t4,5\t0", "13Fields", 0, 0, 0, false},
            {"empty_middle_field\t1\t2\tab c\t100\t+\t0\t0\t1,2,3\t\t123,234\t456,567", "empty_middle_field", 0, 0, 0, false},
            {"trailing_whitespace\t1\t2\tab c\t100\t+\t\t \t", "trailing_whitespace", 1, 2, '+', true},
        }));
    // clang-format on
    CAPTURE(line);

    dorado::alignment::BedFile cut{};
    std::istringstream input_stream{RANDOM_1.bed_line + "\n" + line + "\n" + RANDOM_2.bed_line};

    bool load_result;
    {
        utils::SuppressStdout suppress_error_message{};
        load_result = cut.load(input_stream);
    }
    if (!valid) {
        CHECK_FALSE(load_result);
        return;
    }

    CHECK(load_result);

    const BedFile::Entry EXPECTED{line, start, end, strand};
    CHECK(cut.load(input_stream));
    const auto entries = cut.entries(genome);

    REQUIRE(entries.size() == 1);
    CHECK(entries[0] == EXPECTED);

    auto second_genome_entries = cut.entries("Random");
    REQUIRE(second_genome_entries.size() == 2);
    CHECK(second_genome_entries[0] == RANDOM_1);
    CHECK(second_genome_entries[1] == RANDOM_2);
}

}  // namespace dorado::alignment::bed_file::test