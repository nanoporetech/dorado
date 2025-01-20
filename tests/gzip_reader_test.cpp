#include "utils/gzip_reader.h"

#include "TestUtils.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>

#define CUT_TAG "[dorado::utils::gzip_reader]"
#define DEFINE_TEST(name) CATCH_TEST_CASE(CUT_TAG " " name, CUT_TAG)

namespace dorado::utils::gzip_reader::test {

namespace {
std::filesystem::path get_fastq_folder() { return tests::get_data_dir("fastq"); }
}  // namespace

DEFINE_TEST("constructor with invalid file does not throw") {
    CATCH_REQUIRE_NOTHROW(dorado::utils::GzipReader("invalid_file", 1000));
}

DEFINE_TEST("Reading compressed file matches decompressed file, parameterised.") {
    auto [compressed_file, expected_file] = GENERATE(table<std::string, std::string>({
            {"fastq.fastq.gz", "fastq.fastq"},
            {"fastq_with_us.fastq.gz", "fastq_with_us.fastq"},
            {"fastq_multiple_compressed_sections.fastq.gz",
             "fastq_multiple_compressed_sections_decompressed.fastq"},
    }));
    CATCH_CAPTURE(compressed_file, expected_file);

    GzipReader cut{(get_fastq_folder() / compressed_file).string(), 1000};
    std::ostringstream decompressed_stream{};
    while (cut.read_next() && cut.is_valid()) {
        decompressed_stream << std::string{
                cut.decompressed_buffer().begin(),
                cut.decompressed_buffer().begin() + cut.num_bytes_read()};
    }

    CATCH_CHECK(cut.is_valid());

    std::ifstream fastq_fstream{(get_fastq_folder() / expected_file).string()};
    std::ostringstream expected_stream{};
    expected_stream << fastq_fstream.rdbuf();
    auto expected = expected_stream.str();
    auto actual = decompressed_stream.str();
    CATCH_REQUIRE(expected == actual);
}

}  // namespace dorado::utils::gzip_reader::test
