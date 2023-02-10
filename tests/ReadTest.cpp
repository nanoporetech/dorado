#include "read_pipeline/ReadPipeline.h"
#include "utils/base_mod_utils.h"

#include <catch2/catch.hpp>

#define TEST_GROUP "ReadTest"

TEST_CASE(TEST_GROUP ": Test tag generation", TEST_GROUP) {
    const std::vector<std::string> expected_tags{"qs:i:0",  // empty qstring
                                                 "du:f:1.033000", "ns:i:4132",
                                                 "ts:i:132",      "mx:i:2",
                                                 "ch:i:5",        "st:Z:2017-04-29T09:10:04Z",
                                                 "rn:i:18501",    "f5:Z:batch_0.fast5",
                                                 "sm:f:128.384",  "sd:f:8.258",
                                                 "sv:Z:quantile"};

    dorado::Read test_read;

    test_read.raw_data = torch::empty(4000);
    test_read.sample_rate = 4000.0;
    test_read.shift = 128.3842f;
    test_read.scale = 8.258f;
    test_read.num_trimmed_samples = 132;
    test_read.attributes.mux = 2;
    test_read.attributes.read_number = 18501;
    test_read.attributes.channel_number = 5;
    test_read.attributes.start_time = "2017-04-29T09:10:04Z";
    test_read.attributes.fast5_filename = "batch_0.fast5";

    REQUIRE(test_read.generate_read_tags(false) == expected_tags);
}

TEST_CASE(TEST_GROUP ": Test sam line generation", TEST_GROUP) {
    dorado::Read test_read{};
    SECTION("Generating sam line for empty read throws") {
        REQUIRE_THROWS(test_read.extract_sam_lines(false, false));
    }
    SECTION("Generating sam line for empty seq and qstring throws") {
        test_read.read_id = "test_read";
        REQUIRE_THROWS(test_read.extract_sam_lines(false, false));
    }
    SECTION("Generating sam line for mismatched seq and qstring throws") {
        test_read.read_id = "test_read";
        test_read.seq = "ACGTACGT";
        test_read.qstring = "!!!!";
        REQUIRE_THROWS(test_read.extract_sam_lines(false, false));
    }
    SECTION("Generating sam line for read with non-empty mappings throws") {
        test_read.read_id = "test_read";
        test_read.seq = "ACGTACGT";
        test_read.qstring = "!!!!!!!!";
        test_read.mappings.resize(1);
        REQUIRE_THROWS(test_read.extract_sam_lines(false, false));
    }
    SECTION("Generated sam line for unaligned read is correct") {
        std::vector<std::string> expected_sam_lines{
                "test_read\t4\t*\t0\t0\t*\t*\t0\t8\tACGTACGT\t********\t"
                "qs:i:9\tdu:f:1.033000\tns:i:4132\tts:i:132\tmx:i:2\tch:i:5\tst:Z:2017-04-29T09:10:"
                "04Z\trn:i:"
                "18501\tf5:Z:batch_0.fast5\tsm:f:128.384\tsd:f:8.258\tsv:Z:quantile"};

        test_read.raw_data = torch::empty(4000);
        test_read.sample_rate = 4000.0;
        test_read.shift = 128.3842f;
        test_read.scale = 8.258f;
        test_read.read_id = "test_read";
        test_read.seq = "ACGTACGT";
        test_read.qstring = "********";
        test_read.num_trimmed_samples = 132;
        test_read.attributes.mux = 2;
        test_read.attributes.read_number = 18501;
        test_read.attributes.channel_number = 5;
        test_read.attributes.start_time = "2017-04-29T09:10:04Z";
        test_read.attributes.fast5_filename = "batch_0.fast5";

        REQUIRE(test_read.extract_sam_lines(false, false) == expected_sam_lines);
    }
}

TEST_CASE(TEST_GROUP ": Methylation tag generation", TEST_GROUP) {
    std::string modbase_alphabet = "AXCYGT";
    std::string modbase_long_names = "6mA 5mC";
    std::vector<uint8_t> modbase_probs = {
            235, 20,  0,   0,   0,   0,    // A 6ma (weak call)
            0,   0,   255, 0,   0,   0,    // C
            255, 0,   0,   0,   0,   0,    // A
            0,   0,   0,   0,   255, 0,    // G
            0,   0,   0,   0,   0,   255,  // T
            0,   0,   0,   0,   255, 0,    // G
            1,   254, 0,   0,   0,   0,    // A 6ma
            0,   0,   3,   252, 0,   0,    // C 5ma
            0,   0,   0,   0,   0,   255,  // T
            255, 0,   0,   0,   0,   0,    // A
            255, 0,   0,   0,   0,   0,    // A
            255, 0,   0,   0,   0,   0,    // A
            0,   0,   3,   252, 0,   0,    // C 6ma
            0,   0,   0,   0,   0,   255,  // T
            0,   0,   255, 0,   0,   0,    // C
    };

    dorado::Read read;
    read.seq = "ACAGTGACTAAACTC";
    read.base_mod_probs = modbase_probs;

    std::string methylation_tag;
    SECTION("Methylation threshold is correctly applied") {
        std::string expected_methylation_tag_10_score =
                "MM:Z:A+a,0,1;C+m,1,0;\tML:B:C,20,254,252,252";
        std::string expected_methylation_tag_50_score = "MM:Z:A+a,2;C+m,1,0;\tML:B:C,254,252,252";
        std::string expected_methylation_tag_255_score = "MM:Z:A+a;C+m;\tML:B:C";

        read.base_mod_info = std::make_shared<dorado::utils::BaseModInfo>(modbase_alphabet,
                                                                          modbase_long_names, "");

        // Test generation
        methylation_tag = read.generate_modbase_string(10);
        REQUIRE(methylation_tag == expected_methylation_tag_10_score);

        // Test generation at higher rate excludes the correct mods.
        methylation_tag = read.generate_modbase_string(50);
        REQUIRE(methylation_tag == expected_methylation_tag_50_score);

        // Test generation at max threshold rate excludes everything
        methylation_tag = read.generate_modbase_string(255);
        REQUIRE(methylation_tag == expected_methylation_tag_255_score);
    }

    SECTION("Test generation using CHEBI codes") {
        std::string modbase_long_names_CHEBI = "55555 12345";
        std::string expected_methylation_tag_CHEBI =
                "MM:Z:A+55555,2;C+12345,1,0;\tML:B:C,254,252,252";

        read.base_mod_info = std::make_shared<dorado::utils::BaseModInfo>(
                modbase_alphabet, modbase_long_names_CHEBI, "");
        methylation_tag = read.generate_modbase_string(50);
        REQUIRE(methylation_tag == expected_methylation_tag_CHEBI);
    }

    SECTION("Test generation using AC context for A methylation") {
        std::string context = "XC:_:_:_";
        std::string expected_methylation_tag_with_context =
                "MM:Z:A+a?,0,1,2;C+m,1,0;\tML:B:C,20,254,0,252,252";

        read.base_mod_info = std::make_shared<dorado::utils::BaseModInfo>(
                modbase_alphabet, modbase_long_names, context);

        methylation_tag = read.generate_modbase_string(10);
        REQUIRE(methylation_tag == expected_methylation_tag_with_context);
    }

    SECTION("Test handling of incorrect base names") {
        std::string modbase_long_names_unknown = "12mA 5mq";

        read.base_mod_info = std::make_shared<dorado::utils::BaseModInfo>(
                modbase_alphabet, modbase_long_names_unknown, "");
        methylation_tag = read.generate_modbase_string(50);
        REQUIRE(methylation_tag.empty());
    }
}
