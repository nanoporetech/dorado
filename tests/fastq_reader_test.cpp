#include "utils/fastq_reader.h"

#include <catch2/catch.hpp>

#define CUT_TAG "[dorado::utils::fastq_reader]"
#define DEFINE_TEST(name) TEST_CASE(CUT_TAG " " name, CUT_TAG)

namespace dorado::utils::fastq_reader::test {

namespace {

const std::string VALID_ID_LINE{"@fdbbea47-8893-4055-942b-8c2efe226c17 some description\n"};
const std::string VALID_ID_LINE_2{"@fdbbea47-8893-4055-942b-8c2efe22ffff some other description\n"};
const std::string VALID_SEQ_LINE{"CCCGTTGAAG\n"};
const std::string VALID_SEQ_LINE_WITH_U{"CCCGUUGAAG\n"};
const std::string VALID_SEPARATOR_LINE{"+\n"};
const std::string VALID_QUAL_LINE{"!$#(%(()N~\n"};
const std::string VALID_FASTQ_RECORD{VALID_ID_LINE + VALID_SEQ_LINE + VALID_SEPARATOR_LINE +
                                     VALID_QUAL_LINE};
const std::string VALID_FASTQ_RECORD_2{VALID_ID_LINE_2 + VALID_SEQ_LINE + VALID_SEPARATOR_LINE +
                                       VALID_QUAL_LINE};
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

}  // namespace dorado::utils::fastq_reader::test