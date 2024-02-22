#include "alignment/alignment_processing_items.h"

#include "TestUtils.h"

#include <catch2/catch.hpp>

#include <filesystem>

#define CUT_TAG "[dorado::aligment::cli::AlignmentProcessingItems]"

namespace fs = std::filesystem;
namespace {

const fs::path ROOT_IN_FOLDER{dorado::tests::get_data_dir("alignment_processing_items/input")};
const fs::path ROOT_OUT_FOLDER{ROOT_IN_FOLDER.parent_path() / "output"};
const std::string INPUT_SAM{"sam.sam"};
const std::string INPUT_NOEXT{"no_extension"};
const std::string STDINOUT_INDICATOR{"-"};
}  // namespace

namespace dorado::alignment::cli::test {

TEST_CASE("Constructor with trivial args does not throw", CUT_TAG) {
    CHECK_NOTHROW(AlignmentProcessingItems{"", false, ""});
}

TEST_CASE("initialise with no input reads and recursive flaged returns false", CUT_TAG) {
    AlignmentProcessingItems cut{"", true, ""};
    CHECK_FALSE(cut.initialise());
}

TEST_CASE("initialise with no input and output folder specified returns false", CUT_TAG) {
    AlignmentProcessingItems cut{"", false, ROOT_OUT_FOLDER.string()};
    CHECK_FALSE(cut.initialise());
}

TEST_CASE("initialise with no input and no output folder returns true", CUT_TAG) {
    AlignmentProcessingItems cut{"", false, ""};
    CHECK(cut.initialise());
}

TEST_CASE("get() with no input and no output folder specified returns single item", CUT_TAG) {
    AlignmentProcessingItems cut{"", false, ""};
    cut.initialise();

    CHECK(cut.get().size() == 1);
}

TEST_CASE("get() with no input and no output folder specified returns stdin/stdout", CUT_TAG) {
    AlignmentProcessingItems cut{"", false, ""};
    cut.initialise();

    CHECK(cut.get()[0].input == STDINOUT_INDICATOR);
    CHECK(cut.get()[0].output == STDINOUT_INDICATOR);
}

TEST_CASE("initialise with input file and no output folder returns true", CUT_TAG) {
    AlignmentProcessingItems cut{(ROOT_IN_FOLDER / INPUT_SAM).string(), false, ""};
    CHECK(cut.initialise());
}

TEST_CASE("get() with input file and no output folder returns single item", CUT_TAG) {
    AlignmentProcessingItems cut{(ROOT_IN_FOLDER / INPUT_SAM).string(), false, ""};
    cut.initialise();

    CHECK(cut.get().size() == 1);
}

TEST_CASE("get() with input file and no output folder returns item with correct input", CUT_TAG) {
    const std::string input_file{(ROOT_IN_FOLDER / INPUT_SAM).string()};
    AlignmentProcessingItems cut{input_file, false, ""};
    cut.initialise();

    CHECK(cut.get()[0].input == input_file);
}

TEST_CASE("get() with input file and no output folder returns item with stdout outut", CUT_TAG) {
    AlignmentProcessingItems cut{(ROOT_IN_FOLDER / INPUT_SAM).string(), false, ""};
    cut.initialise();

    CHECK(cut.get()[0].output == STDINOUT_INDICATOR);
}

TEST_CASE("initialise with input file and output folder returns true", CUT_TAG) {
    AlignmentProcessingItems cut{(ROOT_IN_FOLDER / INPUT_SAM).string(), false,
                                 ROOT_OUT_FOLDER.string()};
    CHECK(cut.initialise());
}

TEST_CASE("get() with input file and output folder returns single item", CUT_TAG) {
    AlignmentProcessingItems cut{(ROOT_IN_FOLDER / INPUT_SAM).string(), false,
                                 ROOT_OUT_FOLDER.string()};
    cut.initialise();

    CHECK(cut.get().size() == 1);
}

TEST_CASE("get() with input file and output folder returns item with correct input", CUT_TAG) {
    const std::string input_file{(ROOT_IN_FOLDER / INPUT_SAM).string()};
    AlignmentProcessingItems cut{input_file, false, ROOT_OUT_FOLDER.string()};
    cut.initialise();

    CHECK(cut.get()[0].input == input_file);
}

TEST_CASE("get() with input file and output folder returns output with correct folder", CUT_TAG) {
    const std::string input_file{(ROOT_IN_FOLDER / INPUT_SAM).string()};
    AlignmentProcessingItems cut{input_file, false, ROOT_OUT_FOLDER.string()};
    cut.initialise();

    const std::string expected_output{
            (ROOT_OUT_FOLDER / INPUT_SAM).replace_extension("bam").string()};
    CHECK(cut.get()[0].output == expected_output);
}

TEST_CASE("get() input file with no extension returns output with bam extension", CUT_TAG) {
    const std::string input_file{(ROOT_IN_FOLDER / INPUT_NOEXT).string()};
    AlignmentProcessingItems cut{input_file, false, ROOT_OUT_FOLDER.string()};
    cut.initialise();

    const std::string expected_output{
            (ROOT_OUT_FOLDER / INPUT_NOEXT).replace_extension("bam").string()};
    CHECK(cut.get()[0].output == expected_output);
}

TEST_CASE("initialise() with input folder and no output folder returns false", CUT_TAG) {
    AlignmentProcessingItems cut{ROOT_IN_FOLDER.string(), false, ""};
    CHECK_FALSE(cut.initialise());
}

TEST_CASE(
        "get() with input folder without recursive returns number of files in the root folder only",
        CUT_TAG) {
    AlignmentProcessingItems cut{ROOT_IN_FOLDER.string(), false, ROOT_OUT_FOLDER.string()};
    cut.initialise();

    // bam.bam, fa.fa, fastq.fastq, fq.fq, no_extension, sam.sam, sam_gz.sam.gz, sam_gzip.sam.gzip, no_extension.gz
    std::size_t expected_num_items{9};
    CHECK(cut.get().size() == expected_num_items);
}

TEST_CASE("get() with input folder and recursive returns number of files recursively", CUT_TAG) {
    AlignmentProcessingItems cut{ROOT_IN_FOLDER.string(), true, ROOT_OUT_FOLDER.string()};
    cut.initialise();

    // bam.bam, fa.fa, fastq.fastq, fq.fq, no_extension, sam.sam, sam_gz.sam.gz, sam_gzip.sam.gzip, no_extension.gz
    // dup, dup.bam, dup.fa, dup.fastq, dup.fq, dup.sam
    std::size_t expected_num_items{15};

    CHECK(cut.get().size() == expected_num_items);
}

TEST_CASE("get() with input 'sam_gz.sam.gz' returns output as 'sam_gz.bam'", CUT_TAG) {
    AlignmentProcessingItems cut{(ROOT_IN_FOLDER / "sam_gz.sam.gz").string(), false,
                                 ROOT_OUT_FOLDER.string()};
    cut.initialise();

    CHECK(cut.get()[0].output == (ROOT_OUT_FOLDER / "sam_gz.bam").string());
}
}  // namespace dorado::alignment::cli::test