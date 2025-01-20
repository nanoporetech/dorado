#include "alignment/alignment_processing_items.h"

#include "TestUtils.h"
#include "utils/PostCondition.h"
#include "utils/stream_utils.h"

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <map>

#define CUT_TAG "[dorado::aligment::AlignmentProcessingItems]"

namespace fs = std::filesystem;
namespace {

const fs::path ROOT_IN_FOLDER{dorado::tests::get_data_dir("alignment_processing_items/input")};
const fs::path DUP_FOLDER{ROOT_IN_FOLDER / "duplicates"};
const fs::path OUT_FOLDER{ROOT_IN_FOLDER.parent_path() / "output"};
const std::string INPUT_SAM{"sam.sam"};
const std::string INPUT_NOEXT{"no_extension"};
const std::string NON_HTS_FILE{"non_hts_file.txt"};
const std::string STDINOUT_INDICATOR{"-"};
}  // namespace

namespace dorado::alignment::test {

CATCH_TEST_CASE("Constructor with trivial args does not throw", CUT_TAG) {
    CATCH_CHECK_NOTHROW(AlignmentProcessingItems{"", false, "", false});
}

CATCH_TEST_CASE("initialise with no input reads and recursive flagged returns false", CUT_TAG) {
    dorado::utils::SuppressStdout suppress_error_message{};
    AlignmentProcessingItems cut{"", true, "", false};
    CATCH_CHECK_FALSE(cut.initialise());
}

CATCH_TEST_CASE("initialise with no input and output folder specified returns false", CUT_TAG) {
    AlignmentProcessingItems cut{"", false, OUT_FOLDER.string(), false};
    dorado::utils::SuppressStdout suppress_error_message{};
    CATCH_CHECK_FALSE(cut.initialise());
}

CATCH_TEST_CASE("initialise with no input and no output folder returns true", CUT_TAG) {
    AlignmentProcessingItems cut{"", false, "", false};
    CATCH_CHECK(cut.initialise());
}

CATCH_TEST_CASE("get() with no input and no output folder specified returns single item", CUT_TAG) {
    AlignmentProcessingItems cut{"", false, "", false};
    cut.initialise();

    CATCH_CHECK(cut.get().size() == 1);
}

CATCH_TEST_CASE("get() with no input and no output folder specified returns stdin/stdout",
                CUT_TAG) {
    AlignmentProcessingItems cut{"", false, "", false};
    cut.initialise();

    CATCH_CHECK(cut.get()[0].input == STDINOUT_INDICATOR);
    CATCH_CHECK(cut.get()[0].output == STDINOUT_INDICATOR);
}

CATCH_TEST_CASE("initialise with input file and no output folder returns true", CUT_TAG) {
    AlignmentProcessingItems cut{(ROOT_IN_FOLDER / INPUT_SAM).string(), false, "", false};
    CATCH_CHECK(cut.initialise());
}

CATCH_TEST_CASE("initialise with input file in current directory returns true", CUT_TAG) {
    // Create basic SAM file in a temp directory and change curdir to that temp directory.
    auto tmp_dir = tests::make_temp_dir("aligner_input_from_curdir");
    auto tmp_filename = "empty_file.sam";
    auto tmp_filepath = tmp_dir.m_path / tmp_filename;
    std::ofstream outfile(tmp_filepath.string());

    CATCH_REQUIRE(outfile.is_open());
    outfile << "@HD\tVN:1.6\tSO:unknown\n";
    outfile.close();

    auto orig_cwd = fs::current_path();
    fs::current_path(tmp_dir.m_path);
    auto revert_cwd = utils::PostCondition([&orig_cwd]() { fs::current_path(orig_cwd); });
    AlignmentProcessingItems cut{tmp_filename, false, OUT_FOLDER.string(), false};
    CATCH_CHECK(cut.initialise());
}

CATCH_TEST_CASE("initialise with invalid input file and no output folder returns false", CUT_TAG) {
    AlignmentProcessingItems cut{(ROOT_IN_FOLDER / NON_HTS_FILE).string(), false, "", false};
    CATCH_CHECK_FALSE(cut.initialise());
}

CATCH_TEST_CASE("get() with input file and no output folder returns single item", CUT_TAG) {
    AlignmentProcessingItems cut{(ROOT_IN_FOLDER / INPUT_SAM).string(), false, "", false};
    cut.initialise();

    CATCH_CHECK(cut.get().size() == 1);
}

CATCH_TEST_CASE("get() with input file and no output folder returns item with correct input",
                CUT_TAG) {
    const std::string input_file{(ROOT_IN_FOLDER / INPUT_SAM).string()};
    AlignmentProcessingItems cut{input_file, false, "", false};
    cut.initialise();

    CATCH_CHECK(cut.get()[0].input == input_file);
}

CATCH_TEST_CASE("get() with input file and no output folder returns item with stdout outut",
                CUT_TAG) {
    AlignmentProcessingItems cut{(ROOT_IN_FOLDER / INPUT_SAM).string(), false, "", false};
    cut.initialise();

    CATCH_CHECK(cut.get()[0].output == STDINOUT_INDICATOR);
}

CATCH_TEST_CASE("initialise with input file and output folder returns true", CUT_TAG) {
    AlignmentProcessingItems cut{(ROOT_IN_FOLDER / INPUT_SAM).string(), false, OUT_FOLDER.string(),
                                 false};
    CATCH_CHECK(cut.initialise());
}

CATCH_TEST_CASE("initialise with input file and same output folder returns false", CUT_TAG) {
    AlignmentProcessingItems cut{(ROOT_IN_FOLDER / INPUT_SAM).string(), false,
                                 ROOT_IN_FOLDER.string(), false};
    dorado::utils::SuppressStdout suppress_error_message{};
    CATCH_CHECK_FALSE(cut.initialise());
}

CATCH_TEST_CASE("initialise with invalid input file and output folder returns false", CUT_TAG) {
    AlignmentProcessingItems cut{(ROOT_IN_FOLDER / NON_HTS_FILE).string(), false,
                                 OUT_FOLDER.string(), false};
    CATCH_CHECK_FALSE(cut.initialise());
}

CATCH_TEST_CASE("get() with input file and output folder returns single item", CUT_TAG) {
    AlignmentProcessingItems cut{(ROOT_IN_FOLDER / INPUT_SAM).string(), false, OUT_FOLDER.string(),
                                 false};
    cut.initialise();

    CATCH_CHECK(cut.get().size() == 1);
}

CATCH_TEST_CASE("get() with input file and output folder returns item with correct input",
                CUT_TAG) {
    const std::string input_file{(ROOT_IN_FOLDER / INPUT_SAM).string()};
    AlignmentProcessingItems cut{input_file, false, OUT_FOLDER.string(), false};
    cut.initialise();

    CATCH_CHECK(cut.get()[0].input == input_file);
}

CATCH_TEST_CASE("get() with input file and output folder returns output with correct folder",
                CUT_TAG) {
    const std::string input_file{(ROOT_IN_FOLDER / INPUT_SAM).string()};
    AlignmentProcessingItems cut{input_file, false, OUT_FOLDER.string(), false};
    cut.initialise();

    const std::string expected_output{(OUT_FOLDER / INPUT_SAM).replace_extension("bam").string()};
    CATCH_CHECK(cut.get()[0].output == expected_output);
}

CATCH_TEST_CASE("get() input file with no extension returns output with bam extension", CUT_TAG) {
    const std::string input_file{(ROOT_IN_FOLDER / INPUT_NOEXT).string()};
    AlignmentProcessingItems cut{input_file, false, OUT_FOLDER.string(), false};
    cut.initialise();

    const std::string expected_output{(OUT_FOLDER / INPUT_NOEXT).replace_extension("bam").string()};
    CATCH_CHECK(cut.get()[0].output == expected_output);
}

CATCH_TEST_CASE("initialise() with input folder and no output folder returns false", CUT_TAG) {
    AlignmentProcessingItems cut{ROOT_IN_FOLDER.string(), false, "", false};
    dorado::utils::SuppressStdout suppress_error_message{};
    CATCH_CHECK_FALSE(cut.initialise());
}

CATCH_TEST_CASE("initialise() with input folder and same output folder returns false", CUT_TAG) {
    AlignmentProcessingItems cut{ROOT_IN_FOLDER.string(), false, ROOT_IN_FOLDER.string(), false};
    dorado::utils::SuppressStdout suppress_error_message{};
    CATCH_CHECK_FALSE(cut.initialise());
}

CATCH_TEST_CASE(
        "initialise() with input folder and output folder being an existing subfolder of input "
        "folder returns true",
        CUT_TAG) {
    // N.B. This isn't a requirement, this is just documenting current expected behaviour
    // It may well make sense to prevent any possible inadvertent overwriting of input data.
    AlignmentProcessingItems cut{ROOT_IN_FOLDER.string(), false, DUP_FOLDER.string(), false};
    CATCH_CHECK(cut.initialise());
}

CATCH_TEST_CASE("initialise() with input folder and output folder returns true", CUT_TAG) {
    AlignmentProcessingItems cut{ROOT_IN_FOLDER.string(), false, OUT_FOLDER.string(), false};
    CATCH_CHECK(cut.initialise());
}

CATCH_TEST_CASE(
        "get() with input folder without recursive returns number of files in the root folder only",
        CUT_TAG) {
    AlignmentProcessingItems cut{ROOT_IN_FOLDER.string(), false, OUT_FOLDER.string(), false};
    cut.initialise();

    // bam.bam, fa.fa, fastq.fastq, fq.fq, no_extension, sam.sam, sam_gz.sam.gz, sam_gzip.sam.gzip, no_extension_gz.gz
    // non_hts_file.txt should not be included.
    std::size_t expected_num_items{9};
    CATCH_CHECK(cut.get().size() == expected_num_items);
}

CATCH_TEST_CASE("get() with input folder and recursive returns number of files recursively",
                CUT_TAG) {
    AlignmentProcessingItems cut{ROOT_IN_FOLDER.string(), true, OUT_FOLDER.string(), false};
    cut.initialise();

    // bam.bam, fa.fa, fastq.fastq, fq.fq, no_extension, sam.sam, sam_gz.sam.gz, sam_gzip.sam.gzip, no_extension.gz
    // dup, dup.bam, dup.fa, dup.fastq, dup.fq, dup.sam
    // non_hts_file.txt should not be included.
    std::size_t expected_num_items{15};

    CATCH_CHECK(cut.get().size() == expected_num_items);
}

CATCH_TEST_CASE("get() with input 'sam_gz.sam.gz' returns output as 'sam_gz.bam'", CUT_TAG) {
    AlignmentProcessingItems cut{(ROOT_IN_FOLDER / "sam_gz.sam.gz").string(), false,
                                 OUT_FOLDER.string(), false};
    cut.initialise();

    CATCH_CHECK(cut.get()[0].output == (OUT_FOLDER / "sam_gz.bam").string());
}

CATCH_TEST_CASE(
        "get() with input folder containing duplicate flinemane stems returns output with input "
        "extensions preserved",
        CUT_TAG) {
    std::map<std::string, std::string> expected_output_lut{
            {(DUP_FOLDER / "duplicate").string(), (OUT_FOLDER / "duplicate.bam").string()},
            {(DUP_FOLDER / "duplicate.bam").string(), (OUT_FOLDER / "duplicate.bam.bam").string()},
            {(DUP_FOLDER / "duplicate.fa").string(), (OUT_FOLDER / "duplicate.fa.bam").string()},
            {(DUP_FOLDER / "duplicate.fastq").string(),
             (OUT_FOLDER / "duplicate.fastq.bam").string()},
            {(DUP_FOLDER / "duplicate.fq").string(), (OUT_FOLDER / "duplicate.fq.bam").string()},
            {(DUP_FOLDER / "duplicate.sam").string(), (OUT_FOLDER / "duplicate.sam.bam").string()},
    };

    AlignmentProcessingItems cut{DUP_FOLDER.string(), false, OUT_FOLDER.string(), false};
    cut.initialise();

    auto results = cut.get();

    std::map<std::string, std::string> actual_output_lut{};
    for (const auto& item : results) {
        actual_output_lut[item.input] = item.output;
    }

    CATCH_CHECK(actual_output_lut.size() == expected_output_lut.size());
    for (const auto& expected_in_out : expected_output_lut) {
        CATCH_CHECK(actual_output_lut[expected_in_out.first] == expected_in_out.second);
    }
}

}  // namespace dorado::alignment::test
