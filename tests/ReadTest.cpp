#include "read_pipeline/ReadPipeline.h"

#include <catch2/catch.hpp>

#define TEST_GROUP "ReadTest"

TEST_CASE(TEST_GROUP ": Test tag generation", TEST_GROUP) {
    const std::vector<std::string> expected_tags{
            "qs:i:0",  // empty qstring
            "ns:i:121131", "ts:i:2130",         "mx:i:2", "ch:i:5", "st:Z:2017-04-29T09:10:04Z",
            "rn:i:18501",  "f5:Z:batch_0.fast5"};

    Read test_read;
    test_read.num_samples = 121131;
    test_read.num_trimmed_samples = 2130;
    test_read.attributes.mux = 2;
    test_read.attributes.read_number = 18501;
    test_read.attributes.channel_number = 5;
    test_read.attributes.start_time = "2017-04-29T09:10:04Z";
    test_read.attributes.fast5_filename = "batch_0.fast5";

    REQUIRE(test_read.generate_read_tags() == expected_tags);
}

TEST_CASE(TEST_GROUP ": Test sam line generation", TEST_GROUP) {
    Read test_read{};
    SECTION("Generating sam line for empty read throws") {
        REQUIRE_THROWS(test_read.extract_sam_lines());
    }
    SECTION("Generating sam line for empty seq and qstring throws") {
        test_read.read_id = "test_read";
        REQUIRE_THROWS(test_read.extract_sam_lines());
    }
    SECTION("Generating sam line for mismatched seq and qstring throws") {
        test_read.read_id = "test_read";
        test_read.seq = "ACGTACGT";
        test_read.qstring = "!!!!";
        REQUIRE_THROWS(test_read.extract_sam_lines());
    }
    SECTION("Generating sam line for read with non-empty mappings throws") {
        test_read.read_id = "test_read";
        test_read.seq = "ACGTACGT";
        test_read.qstring = "!!!!!!!!";
        test_read.mappings.resize(1);
        REQUIRE_THROWS(test_read.extract_sam_lines());
    }
    SECTION("Generated sam line for unaligned read is correct") {
        std::vector<std::string> expected_sam_lines{
                "test_read\t4\t*\t0\t0\t*\t*\t0\t8\tACGTACGT\t********\t"
                "qs:i:9\tns:i:121131\tts:i:2130\tmx:i:2\tch:i:5\tst:Z:2017-04-29T09:10:04Z\trn:i:"
                "18501\tf5:Z:batch_0.fast5"};

        test_read.read_id = "test_read";
        test_read.seq = "ACGTACGT";
        test_read.qstring = "********";
        test_read.num_samples = 121131;
        test_read.num_trimmed_samples = 2130;
        test_read.attributes.mux = 2;
        test_read.attributes.read_number = 18501;
        test_read.attributes.channel_number = 5;
        test_read.attributes.start_time = "2017-04-29T09:10:04Z";
        test_read.attributes.fast5_filename = "batch_0.fast5";

        REQUIRE(test_read.extract_sam_lines() == expected_sam_lines);
    }
}
