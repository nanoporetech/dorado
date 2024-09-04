#include "utils/fastq_reader.h"

#include <catch2/catch.hpp>

#include <sstream>

#define CUT_TAG "[dorado::utils::fastq_reader]"
#define DEFINE_TEST(name) TEST_CASE(CUT_TAG " " name, CUT_TAG)

namespace dorado::utils::fastq_reader::test {

namespace {

const std::string VALID_ID{"@fdbbea47-8893-4055-942b-8c2efe226c17 some description"};
const std::string VALID_ID_LINE{VALID_ID + "\n"};
const std::string VALID_ID_2{"@fdbbea47-8893-4055-942b-8c2efe22ffff some other description"};
const std::string VALID_ID_LINE_2{VALID_ID_2 + "\n"};
const std::string VALID_SEQ{"CCCGTTGAAG"};
const std::string VALID_SEQ_LINE{VALID_SEQ + "\n"};
const std::string VALID_SEQ_LINE_WITH_U{"CCCGUUGAAG\n"};
const std::string VALID_SEQ_2{"ACCGTTGCAT"};
const std::string VALID_SEQ_LINE_2{VALID_SEQ_2 + "\n"};
const std::string VALID_SEPARATOR{"+"};
const std::string VALID_SEPARATOR_LINE{VALID_SEPARATOR + "\n"};
const std::string VALID_QUAL{"!$#(%(()N~"};
const std::string VALID_QUAL_LINE{VALID_QUAL + "\n"};
const std::string VALID_QUAL_2{"$(#%(()N~!"};
const std::string VALID_QUAL_LINE_2{VALID_QUAL_2 + "\n"};
const std::string VALID_FASTQ_RECORD{VALID_ID_LINE + VALID_SEQ_LINE + VALID_SEPARATOR_LINE +
                                     VALID_QUAL_LINE};
const std::string VALID_FASTQ_RECORD_2{VALID_ID_LINE_2 + VALID_SEQ_LINE_2 + VALID_SEPARATOR_LINE +
                                       VALID_QUAL_LINE_2};
const std::string VALID_FASTQ_U_RECORD{VALID_ID_LINE + VALID_SEQ_LINE_WITH_U +
                                       VALID_SEPARATOR_LINE + VALID_QUAL_LINE};
const std::string MISSING_QUAL_FIELD_RECORD{VALID_ID_LINE + VALID_SEQ_LINE + VALID_SEPARATOR_LINE};

}  // namespace

DEFINE_TEST("is_fastq with non existent file return false") {
    REQUIRE_FALSE(is_fastq("non_existent_file.278y"));
}

DEFINE_TEST("is_fastq parameterized testing") {
    auto [input_text, is_valid, description] = GENERATE(table<std::string, bool, std::string>({
            {VALID_FASTQ_RECORD + VALID_FASTQ_RECORD_2, true, "valid fastq"},
            {std::string{"\n"} + VALID_SEQ_LINE + VALID_SEPARATOR_LINE + VALID_QUAL_LINE +
                     VALID_FASTQ_RECORD_2,
             false, "empty id line returns false"},

            {VALID_SEQ_LINE + VALID_SEPARATOR_LINE + VALID_QUAL_LINE + VALID_ID_LINE +
                     VALID_FASTQ_RECORD_2,
             false, "missing id line returns false"},

            {std::string{"fdbbea47-8893-4055-942b-8c2efe226c17\n"} + VALID_SEQ_LINE +
                     VALID_SEPARATOR_LINE + VALID_QUAL_LINE,
             false, "id line missing '@' prefix returns false"},

            {std::string{"@\n"} + VALID_SEQ_LINE + VALID_SEPARATOR_LINE + VALID_QUAL_LINE, false,
             "id line with only '@' returns false"},

            {std::string{"@ blah\n"} + VALID_SEQ_LINE + VALID_SEPARATOR_LINE + VALID_QUAL_LINE,
             false, "id line '@ description only' returns false"},

            {VALID_ID_LINE + "\n" + VALID_SEPARATOR_LINE + VALID_QUAL_LINE, false,
             "empty sequence line returns false"},

            {VALID_ID_LINE + "ACGTPCAGTT\n" + VALID_SEPARATOR_LINE + VALID_QUAL_LINE, false,
             "sequence line containing invalid characters returns false"},

            {VALID_ID_LINE + VALID_SEQ_LINE_WITH_U + VALID_SEPARATOR_LINE + VALID_QUAL_LINE, true,
             "sequence line containing Us instead of Ts returns true"},

            {VALID_ID_LINE + "ACGTACGUAC\n" + VALID_SEPARATOR_LINE + VALID_QUAL_LINE, false,
             "sequence line containing Us and Ts returns false"},

            {VALID_ID_LINE + VALID_SEQ_LINE + "\n" + VALID_QUAL_LINE, false,
             "separator line empty - false"},

            {VALID_ID_LINE + VALID_SEQ_LINE + "\n" + VALID_QUAL_LINE + VALID_FASTQ_RECORD_2, false,
             "missing separator line - false"},

            {VALID_ID_LINE + VALID_SEQ_LINE + "+A\n" + VALID_QUAL_LINE, false,
             "separator line with characters after + returns false"},

            {VALID_ID_LINE + VALID_SEQ_LINE + "-\n" + VALID_QUAL_LINE, false,
             "separator line with invalid character returns false"},

            {VALID_ID_LINE + VALID_SEQ_LINE + VALID_SEPARATOR_LINE + "\n" + VALID_FASTQ_RECORD_2,
             false, "empty quality line - false"},

            {VALID_ID_LINE + "\n" + VALID_SEPARATOR_LINE + "\n" + VALID_FASTQ_RECORD_2, false,
             "empty quality and sequence lines - false"},

            {VALID_ID_LINE + VALID_SEQ_LINE + VALID_SEPARATOR_LINE + VALID_FASTQ_RECORD_2, false,
             "missing quality line - false"},

            {VALID_ID_LINE + VALID_SEQ_LINE + VALID_SEPARATOR_LINE + "$$# %(()NS\n", false,
             "quality line with invalid character 0x20 returns false"},

            {VALID_ID_LINE + VALID_SEQ_LINE + VALID_SEPARATOR_LINE + "!$#(%((~~\x7f\n", false,
             "quality line with invalid character 0x7f returns false"},

            {VALID_ID_LINE + "ACGT\n" + VALID_SEPARATOR_LINE + "!$#(%\n", false,
             "quality line different length to sequence returns false"},
    }));
    CAPTURE(description);
    CAPTURE(input_text);
    std::istringstream input_stream{input_text};
    REQUIRE(is_fastq(input_stream) == is_valid);
}

DEFINE_TEST("read_id_view() with valid simple id line returns the read id") {
    FastqRecord cut{};

    REQUIRE(read_id_view("@expected_simple_read_id") == "expected_simple_read_id");
}

DEFINE_TEST("read_id_view() with valid complex id line returns the read id") {
    REQUIRE(read_id_view(
                    "@fdbbea47-8893-4055-942b-8c2efe226c17 "
                    "runid=e2b939f9f7f6b5b78f0b24d0da9da9f6a48d5501 "
                    "sample_id=AMW_RNA_model_training_QC_3 flow_cell_id=FAH44643 ch=258 "
                    "start_time=2017-12-19T08:38:08Z basecall_model_version_id=rna002_70bps_hac@v3 "
                    "basecall_gpu=NVIDIA RTX A5500 Laptop GPU") ==
            "fdbbea47-8893-4055-942b-8c2efe226c17");
}

DEFINE_TEST("FastqReader constructor with invalid file does not throw") {
    REQUIRE_NOTHROW(dorado::utils::FastqReader("invalid_file"));
}

DEFINE_TEST("FastqReader::is_valid constructed with invalid file returns false") {
    dorado::utils::FastqReader cut("invalid_file");
    REQUIRE_FALSE(cut.is_valid());
}

DEFINE_TEST("FastqReader::is_valid constructed with invalid fastq returns false") {
    auto fastq_stream = std::make_unique<std::istringstream>(MISSING_QUAL_FIELD_RECORD);
    dorado::utils::FastqReader cut(std::move(fastq_stream));
    REQUIRE_FALSE(cut.is_valid());
}

DEFINE_TEST("FastqReader::is_valid constructed with valid fastq returns true") {
    auto fastq_stream = std::make_unique<std::istringstream>(VALID_FASTQ_RECORD);
    dorado::utils::FastqReader cut(std::move(fastq_stream));
    REQUIRE(cut.is_valid());
}

DEFINE_TEST("FastqReader::try_get_next_record when not valid returns null") {
    dorado::utils::FastqReader cut("invalid_file");
    auto record = cut.try_get_next_record();
    REQUIRE_FALSE(record.has_value());
}

DEFINE_TEST("FastqReader::try_get_next_record when valid returns expected record") {
    auto fastq_stream = std::make_unique<std::istringstream>(VALID_FASTQ_RECORD);
    dorado::utils::FastqReader cut(std::move(fastq_stream));
    CHECK(cut.is_valid());
    auto record = cut.try_get_next_record();
    REQUIRE(record.has_value());
    CHECK(record->header() == VALID_ID);
    CHECK(record->sequence() == VALID_SEQ);
    CHECK(record->qstring() == VALID_QUAL);
}

DEFINE_TEST("FastqReader::try_get_next_record after returning the only record returns null") {
    auto fastq_stream = std::make_unique<std::istringstream>(VALID_FASTQ_RECORD);
    dorado::utils::FastqReader cut(std::move(fastq_stream));
    auto record = cut.try_get_next_record();
    CHECK(record.has_value());
    record = cut.try_get_next_record();
    REQUIRE_FALSE(record.has_value());
}

DEFINE_TEST("FastqReader::is_valid after try_get_next_record returns null returns false") {
    auto fastq_stream = std::make_unique<std::istringstream>(VALID_FASTQ_RECORD);
    dorado::utils::FastqReader cut(std::move(fastq_stream));
    auto record = cut.try_get_next_record();
    CHECK(record.has_value());
    record = cut.try_get_next_record();
    CHECK_FALSE(record.has_value());
    REQUIRE_FALSE(cut.is_valid());
}

DEFINE_TEST(
        "FastqReader::try_get_next_record after successful try_get_next_record returns next "
        "record") {
    auto fastq_stream =
            std::make_unique<std::istringstream>(VALID_FASTQ_RECORD + VALID_FASTQ_RECORD_2);
    dorado::utils::FastqReader cut(std::move(fastq_stream));
    CHECK(cut.is_valid());
    auto record = cut.try_get_next_record();
    CHECK(record.has_value());
    record = cut.try_get_next_record();
    REQUIRE(record.has_value());
    CHECK(record->header() == VALID_ID_2);
    CHECK(record->sequence() == VALID_SEQ_2);
    CHECK(record->qstring() == VALID_QUAL_2);
}

DEFINE_TEST("FastqReader::try_get_next_record with Us not Ts returns record with Us replaced") {
    auto fastq_stream = std::make_unique<std::istringstream>(VALID_FASTQ_U_RECORD);
    dorado::utils::FastqReader cut(std::move(fastq_stream));
    CHECK(cut.is_valid());
    auto record = cut.try_get_next_record();
    REQUIRE(record.has_value());
    CHECK(record->header() == VALID_ID);
    CHECK(record->sequence() == VALID_SEQ);  // Check Ts not Us
    CHECK(record->qstring() == VALID_QUAL);
}

DEFINE_TEST("run_id_view() parameterized") {
    auto [header_line, expected_run_id] = GENERATE(table<std::string, std::string>({
            {"@read_0 runid=12", "12"},
            {"@read_0 runid=a125g desc", "a125g"},
            {"@read_0 runid=", ""},
            {"@fdbbea47-8893-4055-942b-8c2efe226c17 sample_id=AMW_RNA_model_training_QC_3 "
             "flow_cell_id=FAH44643 ch=258 runid=e2b939f9f7f6b5b78f0b24d0da9da9f6a48d5501 "
             "start_time=2017-12-19T08:38:08Z basecall_model_version_id=rna002_70bps_hac@v3 "
             "basecall_gpu=NVIDIA RTX A5500 Laptop GPU",
             "e2b939f9f7f6b5b78f0b24d0da9da9f6a48d5501"},
    }));
    CAPTURE(header_line);
    REQUIRE(run_id_view(header_line) == expected_run_id);
}

}  // namespace dorado::utils::fastq_reader::test