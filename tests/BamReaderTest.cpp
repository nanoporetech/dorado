#include "MessageSinkUtils.h"
#include "TestUtils.h"
#include "htslib/sam.h"
#include "utils/bam_utils.h"

#include <catch2/catch.hpp>

#include <filesystem>

#define TEST_GROUP "[bam_utils][bam_reader]"

namespace fs = std::filesystem;

TEST_CASE("BamReaderTest: Check reading to sink", TEST_GROUP) {
    fs::path aligner_test_dir = fs::path(get_data_dir("bam_reader"));
    auto fasta = aligner_test_dir / "input.fa";

    // Read FASTA.
    MessageSinkToVector<bam1_t*> sink(100);
    dorado::utils::BamReader reader(fasta.string());
    reader.read(sink, 100);
    auto bam_records = sink.get_messages();
    REQUIRE(bam_records.size() == 10);  // FASTA file has 10 reads.
}

TEST_CASE("BamReaderTest: Check synchronous read", TEST_GROUP) {
    fs::path aligner_test_dir = fs::path(get_data_dir("bam_reader"));
    auto fasta = aligner_test_dir / "input.fa";

    // Read FASTA.
    dorado::utils::BamReader reader(fasta.string());
    uint32_t read_count = 0;
    while (reader.read()) {
        read_count++;
    }
    REQUIRE(read_count == 10);  // FASTA file has 10 reads.
}

TEST_CASE("BamReaderTest: read_bam API", TEST_GROUP) {
    fs::path aligner_test_dir = fs::path(get_data_dir("bam_reader"));
    auto fasta = aligner_test_dir / "input.fa";
    const std::set<std::string> read_ids = {"read_1", "read_2"};

    // Read FASTA.
    auto read_map = dorado::utils::read_bam(fasta.string(), read_ids);
    REQUIRE(read_map.size() == 2);  // read_id filter is only asking for 2 reads.
}
