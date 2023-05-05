#include "MessageSinkUtils.h"
#include "TestUtils.h"
#include "htslib/sam.h"
#include "utils/bam_utils.h"

#include <catch2/catch.hpp>

#include <filesystem>

#define TEST_GROUP "[bam_utils][hts_reader]"

namespace fs = std::filesystem;

TEST_CASE("HtsReaderTest: Read fasta to sink", TEST_GROUP) {
    fs::path aligner_test_dir = fs::path(get_data_dir("bam_reader"));
    auto fasta = aligner_test_dir / "input.fa";

    MessageSinkToVector<dorado::BamPtr> sink(100);
    dorado::utils::HtsReader reader(fasta.string());
    reader.read(sink, 100);
    auto bam_records = sink.get_messages();
    REQUIRE(bam_records.size() == 10);  // FASTA file has 10 reads.
}

TEST_CASE("HtsReaderTest: Read fasta line by line", TEST_GROUP) {
    fs::path aligner_test_dir = fs::path(get_data_dir("bam_reader"));
    auto fasta = aligner_test_dir / "input.fa";

    dorado::utils::HtsReader reader(fasta.string());
    uint32_t read_count = 0;
    while (reader.read()) {
        read_count++;
    }
    REQUIRE(read_count == 10);  // FASTA file has 10 reads.
}

TEST_CASE("HtsReaderTest: read_bam API w/ fasta", TEST_GROUP) {
    fs::path aligner_test_dir = fs::path(get_data_dir("bam_reader"));
    auto fasta = aligner_test_dir / "input.fa";
    const std::set<std::string> read_ids = {"read_1", "read_2"};

    auto read_map = dorado::utils::read_bam(fasta.string(), read_ids);
    REQUIRE(read_map.size() == 2);  // read_id filter is only asking for 2 reads.
}

TEST_CASE("HtsReaderTest: Read SAM to sink", TEST_GROUP) {
    fs::path aligner_test_dir = fs::path(get_data_dir("bam_reader"));
    auto sam = aligner_test_dir / "small.sam";

    MessageSinkToVector<dorado::BamPtr> sink(100);
    dorado::utils::HtsReader reader(sam.string());
    reader.read(sink, 100);
    auto bam_records = sink.get_messages();
    REQUIRE(bam_records.size() == 11);  // SAM file has 11 reads.
}

TEST_CASE("HtsReaderTest: Read SAM line by line", TEST_GROUP) {
    fs::path aligner_test_dir = fs::path(get_data_dir("bam_reader"));
    auto sam = aligner_test_dir / "small.sam";

    dorado::utils::HtsReader reader(sam.string());
    uint32_t read_count = 0;
    while (reader.read()) {
        read_count++;
    }
    REQUIRE(read_count == 11);  // FASTA file has 11 reads.
}

TEST_CASE("HtsReaderTest: read_bam API w/ SAM", TEST_GROUP) {
    fs::path aligner_test_dir = fs::path(get_data_dir("bam_reader"));
    auto sam = aligner_test_dir / "small.sam";
    const std::set<std::string> read_ids = {"d7500028-dfcc-4404-b636-13edae804c55",
                                            "60588a89-f191-414e-b444-ad0815b7d9c9"};

    auto read_map = dorado::utils::read_bam(sam.string(), read_ids);
    REQUIRE(read_map.size() == 2);  // read_id filter is only asking for 2 reads.
}
