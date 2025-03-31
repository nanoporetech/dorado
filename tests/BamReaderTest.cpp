#include "MessageSinkUtils.h"
#include "TestUtils.h"
#include "read_pipeline/HtsReader.h"
#include "utils/bam_utils.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <htslib/sam.h>

#include <filesystem>
#include <unordered_set>

#define TEST_GROUP "[bam_utils][hts_reader]"

namespace fs = std::filesystem;

namespace dorado::hts_reader::test {

CATCH_TEST_CASE("HtsReaderTest: Read fasta to sink", TEST_GROUP) {
    fs::path aligner_test_dir = fs::path(get_data_dir("bam_reader"));
    auto fasta = aligner_test_dir / "input.fa";

    dorado::PipelineDescriptor pipeline_desc;
    std::vector<dorado::Message> bam_records;
    pipeline_desc.add_node<MessageSinkToVector>({}, 100, bam_records);
    auto pipeline = dorado::Pipeline::create(std::move(pipeline_desc), nullptr);

    dorado::HtsReader reader(fasta.string(), std::nullopt);
    reader.read(*pipeline, 100);
    pipeline.reset();
    CATCH_REQUIRE(bam_records.size() == 10);  // FASTA file has 10 reads.
}

CATCH_TEST_CASE("HtsReaderTest: Read fasta line by line", TEST_GROUP) {
    fs::path aligner_test_dir = fs::path(get_data_dir("bam_reader"));
    auto fasta = aligner_test_dir / "input.fa";

    dorado::HtsReader reader(fasta.string(), std::nullopt);
    uint32_t read_count = 0;
    while (reader.read()) {
        read_count++;
    }
    CATCH_REQUIRE(read_count == 10);  // FASTA file has 10 reads.
}

CATCH_TEST_CASE("HtsReaderTest: read_bam API w/ fasta", TEST_GROUP) {
    fs::path aligner_test_dir = fs::path(get_data_dir("bam_reader"));
    auto fasta = aligner_test_dir / "input.fa";
    const std::unordered_set<std::string> read_ids = {"read_1", "read_2"};

    auto read_map = dorado::read_bam(fasta.string(), read_ids);
    CATCH_REQUIRE(read_map.size() == 2);  // read_id filter is only asking for 2 reads.
}

CATCH_TEST_CASE("HtsReaderTest: Read SAM to sink", TEST_GROUP) {
    fs::path aligner_test_dir = fs::path(get_data_dir("bam_reader"));
    auto sam = aligner_test_dir / "small.sam";

    dorado::PipelineDescriptor pipeline_desc;
    std::vector<dorado::Message> bam_records;
    pipeline_desc.add_node<MessageSinkToVector>({}, 100, bam_records);
    auto pipeline = dorado::Pipeline::create(std::move(pipeline_desc), nullptr);

    dorado::HtsReader reader(sam.string(), std::nullopt);
    reader.read(*pipeline, 100);
    pipeline.reset();
    CATCH_REQUIRE(bam_records.size() == 11);  // SAM file has 11 reads.
}

CATCH_TEST_CASE("HtsReaderTest: Read SAM line by line", TEST_GROUP) {
    fs::path aligner_test_dir = fs::path(get_data_dir("bam_reader"));
    auto sam = aligner_test_dir / "small.sam";

    dorado::HtsReader reader(sam.string(), std::nullopt);
    uint32_t read_count = 0;
    while (reader.read()) {
        read_count++;
    }
    CATCH_REQUIRE(read_count == 11);  // FASTA file has 11 reads.
}

CATCH_TEST_CASE("HtsReaderTest: get_tag", TEST_GROUP) {
    fs::path aligner_test_dir = fs::path(get_data_dir("bam_reader"));
    auto sam = aligner_test_dir / "small.sam";

    dorado::HtsReader reader(sam.string(), std::nullopt);
    while (reader.read()) {
        // All records in small.sam have these set.
        CATCH_CHECK(reader.get_tag<int>("XA") == 42);
        CATCH_CHECK(reader.get_tag<std::string>("XB") == "test");
        // Intentionally bad tag to test that missing tags don't return garbage.
        CATCH_CHECK(reader.get_tag<int>("##") == 0);
        CATCH_CHECK(reader.get_tag<float>("##") == 0);
        CATCH_CHECK(reader.get_tag<std::string>("##") == "");
        // Type mismatch doesn't crash.
        CATCH_CHECK(reader.get_tag<int>("XB") == 0);
        CATCH_CHECK(reader.get_tag<float>("XB") == 0);
        CATCH_CHECK(reader.get_tag<std::string>("XA") == "");
    }
}

CATCH_TEST_CASE("HtsReaderTest: read_bam API w/ SAM", TEST_GROUP) {
    fs::path aligner_test_dir = fs::path(get_data_dir("bam_reader"));
    auto sam = aligner_test_dir / "small.sam";
    const std::unordered_set<std::string> read_ids = {"d7500028-dfcc-4404-b636-13edae804c55",
                                                      "60588a89-f191-414e-b444-ad0815b7d9c9"};

    auto read_map = dorado::read_bam(sam.string(), read_ids);
    CATCH_REQUIRE(read_map.size() == 2);  // read_id filter is only asking for 2 reads.
}

CATCH_TEST_CASE("HtsReaderTest: fetch_read_ids API w/ SAM", TEST_GROUP) {
    fs::path aligner_test_dir = fs::path(get_data_dir("bam_reader"));
    auto sam = aligner_test_dir / "small.sam";
    const std::unordered_set<std::string> read_ids = {"d7500028-dfcc-4404-b636-13edae804c55",
                                                      "60588a89-f191-414e-b444-ad0815b7d9c9"};

    auto read_set = dorado::fetch_read_ids(sam.string());
    CATCH_CHECK(read_set.find("d7500028-dfcc-4404-b636-13edae804c55") != read_set.end());
    CATCH_CHECK(read_set.find("60588a89-f191-414e-b444-ad0815b7d9c9") != read_set.end());
}

CATCH_TEST_CASE(
        "HtsReaderTest: read fastq with minKNOW header expect header embedded in bam aux tag 'fq'",
        TEST_GROUP) {
    const fs::path aligner_test_dir = fs::path(get_data_dir("bam_reader"));
    const auto minknow_fastq_file = aligner_test_dir / "fastq_with_minknow_header.fq";
    dorado::HtsReader cut(minknow_fastq_file.string(), std::nullopt);
    CATCH_CHECK(cut.read());
    const auto fq_tag = bam_aux_get(cut.record.get(), "fq");
    CATCH_REQUIRE(fq_tag != nullptr);
    const auto fq_tag_value = bam_aux2Z(fq_tag);
    CATCH_REQUIRE(fq_tag_value != nullptr);
    const std::string fq_header{fq_tag_value};
    CATCH_REQUIRE(
            fq_header ==
            "@c2707254-5445-4cfb-a414-fce1f12b56c0 runid=5c76f4079ee8f04e80b4b8b2c4b677bce7bebb1e "
            "read=1728 ch=332 start_time=2017-06-16T15:31:55Z");
}

CATCH_TEST_CASE("HtsReaderTest: filename tag added if missing", TEST_GROUP) {
    fs::path aligner_test_dir = fs::path(get_data_dir("bam_reader"));
    auto filename = GENERATE("input.fa", "fastq_with_tags.fq");
    auto fasta = aligner_test_dir / filename;

    dorado::HtsReader reader(fasta.string(), std::nullopt);
    while (reader.read()) {
        // All should be given the name of input file.
        CATCH_CHECK(reader.get_tag<std::string>("fn") == filename);
    }
}

}  // namespace dorado::hts_reader::test
