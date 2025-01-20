#include "utils/fasta_reader.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <sstream>

#define CUT_TAG "[dorado::utils::fasta_reader]"
#define DEFINE_TEST(name) CATCH_TEST_CASE(CUT_TAG " " name, CUT_TAG)

namespace dorado::utils::fasta_reader::test {

namespace {

const std::string VALID_ID{">fdbbea47-8893-4055-942b-8c2efe226c17 some description"};
const std::string VALID_ID_LINE{VALID_ID + "\n"};
const std::string VALID_ID_WITH_TABS{
        ">8623ac42-0956-4692-9a93-bdd99bf1a94e\tst:Z:2023-06-22T07:17:48.308+00:00\tRG:Z:"
        "6a94c5e38fbe36232d63fd05555e41368b204cda_dna_r10.4.1_e8.2_400bps_hac@v4.3.0"};
const std::string VALID_ID_LINE_WITH_TABS{VALID_ID_WITH_TABS + "\n"};
const std::string VALID_ID_2{">fdbbea47-8893-4055-942b-8c2efe22ffff some other description"};
const std::string VALID_ID_LINE_2{VALID_ID_2 + "\n"};
const std::string VALID_SEQ{"CCCGTTGAAG"};
const std::string VALID_SEQ_LINE{VALID_SEQ + "\n"};
const std::string VALID_LINE_WRAPPED_SEQ{"CCCGT\nTGAAG"};
const std::string VALID_LINE_WRAPPED_SEQ_LINE{VALID_LINE_WRAPPED_SEQ + "\n"};
const std::string VALID_SEQ_LINE_WITH_U{"CCCGUUGAAG\n"};
const std::string VALID_SEQ_2{"ACCGTTGCAT"};
const std::string VALID_SEQ_LINE_2{VALID_SEQ_2 + "\n"};
const std::string VALID_FASTA_RECORD{VALID_ID_LINE + VALID_SEQ_LINE};
const std::string VALID_FASTA_RECORD_2{VALID_ID_LINE_2 + VALID_SEQ_LINE_2};
const std::string VALID_FASTA_U_RECORD{VALID_ID_LINE + VALID_SEQ_LINE_WITH_U};

}  // namespace

DEFINE_TEST("is_fasta with non existent file return false") {
    CATCH_REQUIRE_FALSE(is_fasta("non_existent_file.278y"));
}

DEFINE_TEST("is_fasta parameterized testing") {
    auto [input_text, is_valid, description] = GENERATE(table<std::string, bool, std::string>({
            {VALID_FASTA_RECORD + VALID_FASTA_RECORD_2, true, "valid fasta"},

            {"", false, "empty input returns false"},

            {std::string{"\n"} + VALID_SEQ_LINE + VALID_FASTA_RECORD_2, false,
             "empty id line returns false"},

            {VALID_SEQ_LINE + VALID_ID_LINE + VALID_FASTA_RECORD_2, false,
             "missing id line returns false"},

            {VALID_ID_LINE_WITH_TABS + VALID_SEQ_LINE, true,
             "valid header with tabbed aux fields returns true"},

            {std::string{"fdbbea47-8893-4055-942b-8c2efe226c17\n"} + VALID_SEQ_LINE, false,
             "id line missing '>' prefix returns false"},

            {std::string{">\n"} + VALID_SEQ_LINE, false, "id line with only '>' returns false"},

            {std::string{"> blah\n"} + VALID_SEQ_LINE, false,
             "id line '> description only' returns false"},

            {VALID_ID_LINE + "\n", false, "empty sequence line returns false"},

            {VALID_ID_LINE + "ACGTPCAGTT\n", false,
             "sequence line containing invalid characters returns false"},

            {VALID_ID_LINE + VALID_SEQ_LINE_WITH_U, true,
             "sequence line containing Us instead of Ts returns true"},

            {VALID_ID_LINE + "ACGTACGUAC\n", false,
             "sequence line containing Us and Ts returns false"},

            {VALID_ID_LINE + VALID_LINE_WRAPPED_SEQ_LINE, true,
             "valid record with line wrapped sequence returns true"},

    }));
    CATCH_CAPTURE(description);
    CATCH_CAPTURE(input_text);
    std::istringstream input_stream{input_text};
    CATCH_REQUIRE(is_fasta(input_stream) == is_valid);
}

DEFINE_TEST("FastaRecord::record_name() parameterized") {
    auto [header_line, expected_record_name] = GENERATE(table<std::string, std::string>({
            {">expected_simple_record_name", "expected_simple_record_name"},
            {">8623ac42-0956\tst:Z:2023-06-22T07\tRG:Z:6a94c5e3.0", "8623ac42-0956"},
            {">read_0 runid=1", "read_0"},
    }));
    CATCH_CAPTURE(header_line);
    FastaRecord cut{};
    cut.set_header(std::move(header_line));

    CATCH_REQUIRE(cut.record_name() == expected_record_name);
}

DEFINE_TEST("FastaRecord::get_tokens() with no descroption returns only the record id") {
    FastaRecord cut{};
    cut.set_header(">read_0");
    const auto tokens = cut.get_tokens();
    CATCH_REQUIRE(tokens.size() == 1);
    CATCH_CHECK(tokens[0] == ">read_0");
}

DEFINE_TEST("FastaRecord::get_tokens() with minKNOW style header returns tokens") {
    FastaRecord cut{};
    cut.set_header(
            ">c2707254-5445-4cfb-a414-fce1f12b56c0 runid=5c76f4079ee8f04e80b4b8b2c4b677bce7bebb1e "
            "read=1728 ch=332 start_time=2017-06-16T15:31:55Z");

    const auto tokens = cut.get_tokens();
    CATCH_REQUIRE(tokens.size() == 5);
    CATCH_CHECK(tokens[0] == ">c2707254-5445-4cfb-a414-fce1f12b56c0");
    CATCH_CHECK(tokens[1] == "runid=5c76f4079ee8f04e80b4b8b2c4b677bce7bebb1e");
    CATCH_CHECK(tokens[2] == "read=1728");
    CATCH_CHECK(tokens[3] == "ch=332");
    CATCH_CHECK(tokens[4] == "start_time=2017-06-16T15:31:55Z");
}

DEFINE_TEST("FastaqRecord::get_tokens() with single BAM tag returns that tag") {
    FastaRecord cut{};
    cut.set_header(">read_0\tRG:Z:6a94c5e3");

    const auto tokens = cut.get_tokens();
    CATCH_REQUIRE(tokens.size() == 2);
    CATCH_CHECK(tokens[0] == ">read_0");
    CATCH_CHECK(tokens[1] == "RG:Z:6a94c5e3");
}

DEFINE_TEST("FastaRecord::get_tokens() with two BAM tags containing spaces returns both tags") {
    FastaRecord cut{};
    cut.set_header(">read_0\tfq:Z:some text field\tRG:Z:6a94c5e3");

    const auto tokens = cut.get_tokens();
    CATCH_REQUIRE(tokens.size() == 3);
    CATCH_CHECK(tokens[0] == ">read_0");
    CATCH_CHECK(tokens[1] == "fq:Z:some text field");
    CATCH_CHECK(tokens[2] == "RG:Z:6a94c5e3");
}

DEFINE_TEST("FastaReader constructor with invalid file does not throw") {
    CATCH_REQUIRE_NOTHROW(dorado::utils::FastaReader("invalid_file"));
}

DEFINE_TEST("FastaReader::is_valid constructed with invalid file returns false") {
    dorado::utils::FastaReader cut("invalid_file");
    CATCH_REQUIRE_FALSE(cut.is_valid());
}

DEFINE_TEST("FastaReader::is_valid constructed with invalid fasta returns false") {
    auto fasta_stream = std::make_unique<std::istringstream>(">name\n");
    dorado::utils::FastaReader cut(std::move(fasta_stream));
    CATCH_REQUIRE_FALSE(cut.is_valid());
}

DEFINE_TEST("FastaReader::is_valid constructed with valid fasta returns true") {
    auto fasta_stream = std::make_unique<std::istringstream>(VALID_FASTA_RECORD);
    dorado::utils::FastaReader cut(std::move(fasta_stream));
    CATCH_REQUIRE(cut.is_valid());
}

DEFINE_TEST("FastaReader::try_get_next_record when not valid returns null") {
    dorado::utils::FastaReader cut("invalid_file");
    auto record = cut.try_get_next_record();
    CATCH_REQUIRE_FALSE(record.has_value());
}

DEFINE_TEST("FastaReader::try_get_next_record when valid returns expected record") {
    auto fasta_stream = std::make_unique<std::istringstream>(VALID_FASTA_RECORD);
    dorado::utils::FastaReader cut(std::move(fasta_stream));
    CATCH_CHECK(cut.is_valid());
    auto record = cut.try_get_next_record();
    CATCH_REQUIRE(record.has_value());
    CATCH_CHECK(record->header() == VALID_ID);
    CATCH_CHECK(record->sequence() == VALID_SEQ);
}

DEFINE_TEST("FastaReader::try_get_next_record after returning the only record returns null") {
    auto fasta_stream = std::make_unique<std::istringstream>(VALID_FASTA_RECORD);
    dorado::utils::FastaReader cut(std::move(fasta_stream));
    auto record = cut.try_get_next_record();
    CATCH_CHECK(record.has_value());
    record = cut.try_get_next_record();
    CATCH_REQUIRE_FALSE(record.has_value());
}

DEFINE_TEST("FastaReader::is_valid after try_get_next_record returns null returns false") {
    auto fasta_stream = std::make_unique<std::istringstream>(VALID_FASTA_RECORD);
    dorado::utils::FastaReader cut(std::move(fasta_stream));
    auto record = cut.try_get_next_record();
    CATCH_CHECK(record.has_value());
    record = cut.try_get_next_record();
    CATCH_CHECK_FALSE(record.has_value());
    CATCH_REQUIRE_FALSE(cut.is_valid());
}

DEFINE_TEST(
        "FastaReader::try_get_next_record after successful try_get_next_record returns next "
        "record") {
    auto fasta_stream =
            std::make_unique<std::istringstream>(VALID_FASTA_RECORD + VALID_FASTA_RECORD_2);
    dorado::utils::FastaReader cut(std::move(fasta_stream));
    CATCH_CHECK(cut.is_valid());
    auto record = cut.try_get_next_record();
    CATCH_CHECK(record.has_value());
    record = cut.try_get_next_record();
    CATCH_REQUIRE(record.has_value());
    CATCH_CHECK(record->header() == VALID_ID_2);
    CATCH_CHECK(record->sequence() == VALID_SEQ_2);
}

DEFINE_TEST("FastaReader::try_get_next_record with Us not Ts returns record with Us replaced") {
    auto fasta_stream = std::make_unique<std::istringstream>(VALID_FASTA_U_RECORD);
    dorado::utils::FastaReader cut(std::move(fasta_stream));
    CATCH_CHECK(cut.is_valid());
    auto record = cut.try_get_next_record();
    CATCH_REQUIRE(record.has_value());
    CATCH_CHECK(record->header() == VALID_ID);
    CATCH_CHECK(record->sequence() == VALID_SEQ);  // Check Ts not Us
}

}  // namespace dorado::utils::fasta_reader::test
