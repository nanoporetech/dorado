#include "utils/fastq_reader.h"

#include "TestUtils.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <filesystem>
#include <sstream>

#define CUT_TAG "[dorado::utils::fastq_reader]"
#define DEFINE_TEST(name) CATCH_TEST_CASE(CUT_TAG " " name, CUT_TAG)

namespace dorado::utils::fastq_reader::test {

namespace {

const std::string VALID_ID{"@fdbbea47-8893-4055-942b-8c2efe226c17 some description"};
const std::string VALID_ID_LINE{VALID_ID + "\n"};
const std::string VALID_ID_WITH_TABS{
        "@8623ac42-0956-4692-9a93-bdd99bf1a94e\tst:Z:2023-06-22T07:17:48.308+00:00\tRG:Z:"
        "6a94c5e38fbe36232d63fd05555e41368b204cda_dna_r10.4.1_e8.2_400bps_hac@v4.3.0"};
const std::string VALID_ID_LINE_WITH_TABS{VALID_ID_WITH_TABS + "\n"};
const std::string VALID_ID_2{"@fdbbea47-8893-4055-942b-8c2efe22ffff some other description"};
const std::string VALID_ID_LINE_2{VALID_ID_2 + "\n"};
const std::string VALID_SEQ{"CCCGTTGAAG"};
const std::string VALID_SEQ_LINE{VALID_SEQ + "\n"};
const std::string VALID_LINE_WRAPPED_SEQ{"CCCGT\nTGAAG"};
const std::string VALID_LINE_WRAPPED_SEQ_LINE{VALID_LINE_WRAPPED_SEQ + "\n"};
const std::string VALID_SEQ_LINE_WITH_U{"CCCGUUGAAG\n"};
const std::string VALID_SEQ_2{"ACCGTTGCAT"};
const std::string VALID_SEQ_LINE_2{VALID_SEQ_2 + "\n"};
const std::string VALID_SEPARATOR{"+"};
const std::string VALID_SEPARATOR_LINE{VALID_SEPARATOR + "\n"};
const std::string VALID_QUAL{"!$#(%(()N~"};
const std::string VALID_QUAL_LINE{VALID_QUAL + "\n"};
const std::string VALID_LINE_WRAPPED_QUAL{"!$#(\n%(()N~"};
const std::string VALID_LINE_WRAPPED_QUAL_LINE{VALID_LINE_WRAPPED_QUAL + "\n"};
const std::string VALID_QUAL_2{"$(#%(()N~!"};
const std::string VALID_QUAL_LINE_2{VALID_QUAL_2 + "\n"};
const std::string VALID_FASTQ_RECORD{VALID_ID_LINE + VALID_SEQ_LINE + VALID_SEPARATOR_LINE +
                                     VALID_QUAL_LINE};
const std::string VALID_FASTQ_RECORD_2{VALID_ID_LINE_2 + VALID_SEQ_LINE_2 + VALID_SEPARATOR_LINE +
                                       VALID_QUAL_LINE_2};
const std::string VALID_FASTQ_U_RECORD{VALID_ID_LINE + VALID_SEQ_LINE_WITH_U +
                                       VALID_SEPARATOR_LINE + VALID_QUAL_LINE};
const std::string MISSING_QUAL_FIELD_RECORD{VALID_ID_LINE + VALID_SEQ_LINE + VALID_SEPARATOR_LINE};

std::filesystem::path get_fastq_folder() { return get_data_dir("fastq"); }

}  // namespace

DEFINE_TEST("is_fastq with non existent file return false") {
    CATCH_REQUIRE_FALSE(is_fastq("non_existent_file.278y"));
}

DEFINE_TEST("is_fastq with valid fastq file return true") {
    auto fastq_file = get_fastq_folder() / "fastq.fastq";
    CATCH_REQUIRE(is_fastq(fastq_file.string()));
}

DEFINE_TEST("is_fastq with valid compressed fastq file return true") {
    auto compressed_fastq_file = get_fastq_folder() / "fastq.fastq.gz";
    CATCH_REQUIRE(is_fastq(compressed_fastq_file.string()));
}

DEFINE_TEST("is_fastq with gzip fle containing multiple compressed sections returns true") {
    auto compressed_fastq_file = get_fastq_folder() / "fastq_multiple_compressed_sections.fastq.gz";
    CATCH_REQUIRE(is_fastq(compressed_fastq_file.string()));
}

DEFINE_TEST("is_fastq parameterized testing") {
    auto [input_text, is_valid, description] = GENERATE(table<std::string, bool, std::string>({
            {VALID_FASTQ_RECORD + VALID_FASTQ_RECORD_2, true, "valid fastq"},

            {"", false, "empty input returns false"},

            {std::string{"\n"} + VALID_SEQ_LINE + VALID_SEPARATOR_LINE + VALID_QUAL_LINE +
                     VALID_FASTQ_RECORD_2,
             false, "empty id line returns false"},

            {VALID_SEQ_LINE + VALID_SEPARATOR_LINE + VALID_QUAL_LINE + VALID_ID_LINE +
                     VALID_FASTQ_RECORD_2,
             false, "missing id line returns false"},

            {VALID_ID_LINE_WITH_TABS + VALID_SEQ_LINE + VALID_SEPARATOR_LINE + VALID_QUAL_LINE,
             true, "valid header with tabbed aux fields returns true"},

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

            {VALID_ID_LINE + VALID_SEQ_LINE + "-\n" + VALID_QUAL_LINE, false,
             "separator line with invalid character returns false"},

            {VALID_ID_LINE + VALID_SEQ_LINE + "+" + VALID_ID_LINE.substr(1) + VALID_QUAL_LINE, true,
             "extended separator line with description like header returns true"},

            {VALID_ID_LINE_WITH_TABS + VALID_SEQ_LINE + "+" + VALID_ID_LINE_WITH_TABS.substr(1) +
                     VALID_QUAL_LINE,
             true, "extended separator line with tabbed description like header returns true"},

            {VALID_ID_LINE + VALID_SEQ_LINE + "+ blah\n" + VALID_QUAL_LINE, true,
             "extended separator line with description containing leading SPACE returns true"},

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

            {VALID_ID_LINE + VALID_LINE_WRAPPED_SEQ_LINE + VALID_SEPARATOR_LINE + VALID_QUAL_LINE,
             true, "valid record with line wrapped sequence returns true"},

            {VALID_ID_LINE + "TCA\n\n" + VALID_SEPARATOR_LINE + "ABC\n", false,
             "record with line wrapped sequence containing trailing empty line returns false"},

            {VALID_ID_LINE + "TCA\n" + VALID_SEPARATOR_LINE + "AB\nC\n", true,
             "record with line wrapped qstring returns true"},

            {VALID_ID_LINE + "TCA\n" + VALID_SEPARATOR_LINE + "AB\n\nC\n", false,
             "record with line wrapped qstring containing empty line returns false"},

            {VALID_ID_LINE + "TCA\n" + VALID_SEPARATOR_LINE + "A\n@\nC\n", true,
             "record with valid qstring containing '@' returns true"},

            {VALID_ID_LINE + "ACGTACG\nTAC\nG\nTAC\n" + VALID_SEPARATOR_LINE +
                     "@read_0\nACT\n+\n!$#\n",
             true, "record with qstring equivalent to a valid fastq record returns true"},
    }));
    CATCH_CAPTURE(description);
    CATCH_CAPTURE(input_text);
    std::istringstream input_stream{input_text};
    CATCH_REQUIRE(is_fastq(input_stream) == is_valid);
}

DEFINE_TEST("FastqRecord::read_id_view() parameterized") {
    auto [header_line, expected_read_id] = GENERATE(table<std::string, std::string>({
            {"@expected_simple_read_id", "expected_simple_read_id"},
            {"@8623ac42-0956\tst:Z:2023-06-22T07\tRG:Z:6a94c5e3.0", "8623ac42-0956"},
            {"@read_0 runid=1", "read_0"},
    }));
    CATCH_CAPTURE(header_line);
    FastqRecord cut{};
    cut.set_header(std::move(header_line));

    CATCH_REQUIRE(cut.read_id_view() == expected_read_id);
}

DEFINE_TEST("FastqRecord::run_id_view() parameterized") {
    auto [header_line, expected_run_id] = GENERATE(table<std::string, std::string>({
            {"@read_0 runid=12", "12"},
            {"@read_0 runid=a125g desc", "a125g"},
            {"@read_0 runid=", ""},
            {"@r0\tfq:Z:bam tag contains token runid=a125g\tRG:Z:6a94c5e3", ""},
            {"@fdbbea47-8893-4055-942b-8c2efe226c17 sample_id=AMW_RNA_model_training_QC_3 "
             "flow_cell_id=FAH44643 ch=258 runid=e2b939f9f7f6b5b78f0b24d0da9da9f6a48d5501 "
             "start_time=2017-12-19T08:38:08Z basecall_model_version_id=rna002_70bps_hac@v3 "
             "basecall_gpu=NVIDIA RTX A5500 Laptop GPU",
             "e2b939f9f7f6b5b78f0b24d0da9da9f6a48d5501"},
    }));
    CATCH_CAPTURE(header_line);
    FastqRecord cut{};
    cut.set_header(std::move(header_line));

    CATCH_REQUIRE(cut.run_id_view() == expected_run_id);
}

DEFINE_TEST("FastqRecord::get_bam_tags() with no descroption returns empty") {
    FastqRecord cut{};
    cut.set_header("@read_0");

    CATCH_REQUIRE(cut.get_bam_tags().empty());
}

DEFINE_TEST("FastqRecord::get_bam_tags() with minKNOW style header returns empty") {
    FastqRecord cut{};
    cut.set_header(
            "@c2707254-5445-4cfb-a414-fce1f12b56c0 runid=5c76f4079ee8f04e80b4b8b2c4b677bce7bebb1e "
            "read=1728 ch=332 start_time=2017-06-16T15:31:55Z");

    CATCH_REQUIRE(cut.get_bam_tags().empty());
}

DEFINE_TEST("FastqRecord::get_bam_tags() with single tag returns that tag") {
    FastqRecord cut{};
    cut.set_header("@read_0\tRG:Z:6a94c5e3");

    const auto bam_tags = cut.get_bam_tags();

    CATCH_REQUIRE(bam_tags.size() == 1);
    CATCH_REQUIRE(bam_tags[0] == "RG:Z:6a94c5e3");
}

DEFINE_TEST("FastqRecord::get_bam_tags() with two tags containing spaces returns both tags") {
    FastqRecord cut{};
    cut.set_header("@read_0\tfq:Z:some text field\tRG:Z:6a94c5e3");

    const auto bam_tags = cut.get_bam_tags();

    CATCH_REQUIRE(bam_tags.size() == 2);
    CATCH_REQUIRE(bam_tags[0] == "fq:Z:some text field");
    CATCH_REQUIRE(bam_tags[1] == "RG:Z:6a94c5e3");
}

DEFINE_TEST("FastqReader constructor with invalid file does not throw") {
    CATCH_REQUIRE_NOTHROW(dorado::utils::FastqReader("invalid_file"));
}

DEFINE_TEST("FastqReader::is_valid constructed with invalid file returns false") {
    dorado::utils::FastqReader cut("invalid_file");
    CATCH_REQUIRE_FALSE(cut.is_valid());
}

DEFINE_TEST("FastqReader::is_valid constructed with invalid fastq returns false") {
    auto fastq_stream = std::make_unique<std::istringstream>(MISSING_QUAL_FIELD_RECORD);
    dorado::utils::FastqReader cut(std::move(fastq_stream));
    CATCH_REQUIRE_FALSE(cut.is_valid());
}

DEFINE_TEST("FastqReader::is_valid constructed with valid fastq returns true") {
    auto fastq_stream = std::make_unique<std::istringstream>(VALID_FASTQ_RECORD);
    dorado::utils::FastqReader cut(std::move(fastq_stream));
    CATCH_REQUIRE(cut.is_valid());
}

DEFINE_TEST("FastqReader::is_valid constructed with valid fastq file returns true") {
    auto input_fastq = get_fastq_folder() / "fastq.fastq";
    dorado::utils::FastqReader cut(input_fastq.string());
    CATCH_REQUIRE(cut.is_valid());
}

DEFINE_TEST("FastqReader::is_valid constructed with valid compressed fastq file returns true") {
    auto input_fastq = get_fastq_folder() / "fastq.fastq.gz";
    dorado::utils::FastqReader cut(input_fastq.string());
    CATCH_REQUIRE(cut.is_valid());
}

DEFINE_TEST("FastqReader::try_get_next_record when not valid returns null") {
    dorado::utils::FastqReader cut("invalid_file");
    auto record = cut.try_get_next_record();
    CATCH_REQUIRE_FALSE(record.has_value());
}

DEFINE_TEST("FastqReader::try_get_next_record when valid returns expected record") {
    auto fastq_stream = std::make_unique<std::istringstream>(VALID_FASTQ_RECORD);
    dorado::utils::FastqReader cut(std::move(fastq_stream));
    CATCH_CHECK(cut.is_valid());
    auto record = cut.try_get_next_record();
    CATCH_REQUIRE(record.has_value());
    CATCH_CHECK(record->header() == VALID_ID);
    CATCH_CHECK(record->sequence() == VALID_SEQ);
    CATCH_CHECK(record->qstring() == VALID_QUAL);
}

DEFINE_TEST("FastqReader::try_get_next_record after returning the only record returns null") {
    auto fastq_stream = std::make_unique<std::istringstream>(VALID_FASTQ_RECORD);
    dorado::utils::FastqReader cut(std::move(fastq_stream));
    auto record = cut.try_get_next_record();
    CATCH_CHECK(record.has_value());
    record = cut.try_get_next_record();
    CATCH_REQUIRE_FALSE(record.has_value());
}

DEFINE_TEST("FastqReader::is_valid after try_get_next_record returns null returns false") {
    auto fastq_stream = std::make_unique<std::istringstream>(VALID_FASTQ_RECORD);
    dorado::utils::FastqReader cut(std::move(fastq_stream));
    auto record = cut.try_get_next_record();
    CATCH_CHECK(record.has_value());
    record = cut.try_get_next_record();
    CATCH_CHECK_FALSE(record.has_value());
    CATCH_REQUIRE_FALSE(cut.is_valid());
}

DEFINE_TEST(
        "FastqReader::try_get_next_record after successful try_get_next_record returns next "
        "record") {
    auto fastq_stream =
            std::make_unique<std::istringstream>(VALID_FASTQ_RECORD + VALID_FASTQ_RECORD_2);
    dorado::utils::FastqReader cut(std::move(fastq_stream));
    CATCH_CHECK(cut.is_valid());
    auto record = cut.try_get_next_record();
    CATCH_CHECK(record.has_value());
    record = cut.try_get_next_record();
    CATCH_REQUIRE(record.has_value());
    CATCH_CHECK(record->header() == VALID_ID_2);
    CATCH_CHECK(record->sequence() == VALID_SEQ_2);
    CATCH_CHECK(record->qstring() == VALID_QUAL_2);
}

DEFINE_TEST("FastqReader::try_get_next_record with Us not Ts returns record with Us replaced") {
    auto fastq_stream = std::make_unique<std::istringstream>(VALID_FASTQ_U_RECORD);
    dorado::utils::FastqReader cut(std::move(fastq_stream));
    CATCH_CHECK(cut.is_valid());
    auto record = cut.try_get_next_record();
    CATCH_REQUIRE(record.has_value());
    CATCH_CHECK(record->header() == VALID_ID);
    CATCH_CHECK(record->sequence() == VALID_SEQ);  // Check Ts not Us
    CATCH_CHECK(record->qstring() == VALID_QUAL);
}

DEFINE_TEST("FastqReader files parameterised testing") {
    auto [compressed_file, expected_results_file] = GENERATE(table<std::string, std::string>({
            {"fastq.fastq.gz", "fastq.fastq"},
            {"fastq_with_us.fastq", "fastq.fastq"},
            {"fastq_with_us.fastq.gz", "fastq.fastq"},
            {"fastq_multiple_compressed_sections.fastq.gz",
             "fastq_multiple_compressed_sections_decompressed.fastq"},
    }));
    CATCH_CAPTURE(compressed_file, expected_results_file);

    compressed_file = (get_fastq_folder() / compressed_file).string();
    std::vector<FastqRecord> decompressed_fastq_records{};
    FastqReader compressed_read(compressed_file);
    for (auto record = compressed_read.try_get_next_record(); record;
         record = compressed_read.try_get_next_record()) {
        decompressed_fastq_records.push_back(std::move(*record));
    }

    expected_results_file = (get_fastq_folder() / expected_results_file).string();
    std::vector<FastqRecord> expected_fastq_records{};
    FastqReader expected_results_reader(expected_results_file);
    for (auto record = expected_results_reader.try_get_next_record(); record;
         record = expected_results_reader.try_get_next_record()) {
        expected_fastq_records.push_back(std::move(*record));
    }

    CATCH_REQUIRE(decompressed_fastq_records.size() == expected_fastq_records.size());
    for (std::size_t index{}; index < expected_fastq_records.size(); ++index) {
        CATCH_REQUIRE(decompressed_fastq_records[index] == expected_fastq_records[index]);
    }
}

}  // namespace dorado::utils::fastq_reader::test
