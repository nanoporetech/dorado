#include "htslib/sam.h"
#include "read_pipeline/ReadPipeline.h"
#include "utils/base_mod_utils.h"

#include <catch2/catch.hpp>

#define TEST_GROUP "ReadTest"

using Catch::Matchers::Equals;

TEST_CASE(TEST_GROUP ": Test tag generation", TEST_GROUP) {
    dorado::Read test_read;
    test_read.read_id = "read1";
    test_read.raw_data = torch::empty(4000);
    test_read.seq = "ACGT";
    test_read.qstring = "////";
    test_read.sample_rate = 4000.0;
    test_read.shift = 128.3842f;
    test_read.scale = 8.258f;
    test_read.scaling_method = "quantile";
    test_read.num_trimmed_samples = 132;
    test_read.attributes.mux = 2;
    test_read.attributes.read_number = 18501;
    test_read.attributes.channel_number = 5;
    test_read.attributes.start_time = "2017-04-29T09:10:04Z";
    test_read.attributes.fast5_filename = "batch_0.fast5";
    test_read.run_id = "xyz";
    test_read.model_name = "test_model";
    test_read.is_duplex = false;

    auto alignments = test_read.extract_sam_lines(false);
    REQUIRE(alignments.size() == 1);
    bam1_t* aln = alignments[0].get();

    CHECK(bam_aux2i(bam_aux_get(aln, "qs")) == 14);
    CHECK(bam_aux2i(bam_aux_get(aln, "ns")) == 4132);
    CHECK(bam_aux2i(bam_aux_get(aln, "ts")) == 132);
    CHECK(bam_aux2i(bam_aux_get(aln, "mx")) == 2);
    CHECK(bam_aux2i(bam_aux_get(aln, "ch")) == 5);
    CHECK(bam_aux2i(bam_aux_get(aln, "rn")) == 18501);
    CHECK(bam_aux2i(bam_aux_get(aln, "rn")) == 18501);
    CHECK(bam_aux2i(bam_aux_get(aln, "dx")) == 0);

    CHECK(bam_aux2f(bam_aux_get(aln, "du")) == Approx(1.033).margin(1e-6));
    CHECK(bam_aux2f(bam_aux_get(aln, "sm")) == 128.3842f);
    CHECK(bam_aux2f(bam_aux_get(aln, "sd")) == 8.258f);

    CHECK_THAT(bam_aux2Z(bam_aux_get(aln, "st")), Equals("2017-04-29T09:10:04Z"));
    CHECK_THAT(bam_aux2Z(bam_aux_get(aln, "fn")), Equals("batch_0.fast5"));
    CHECK_THAT(bam_aux2Z(bam_aux_get(aln, "sv")), Equals("quantile"));
    CHECK_THAT(bam_aux2Z(bam_aux_get(aln, "RG")), Equals("xyz_test_model"));

    // Update read to be duplex
    test_read.is_duplex = true;

    alignments = test_read.extract_sam_lines(false);
    REQUIRE(alignments.size() == 1);
    aln = alignments[0].get();

    CHECK(bam_aux2i(bam_aux_get(aln, "dx")) == 1);
    CHECK_THAT(bam_aux2Z(bam_aux_get(aln, "RG")), Equals("xyz_test_model"));
}

TEST_CASE(TEST_GROUP ": Test sam record generation", TEST_GROUP) {
    dorado::Read test_read{};
    SECTION("Generating sam record for empty read throws") {
        REQUIRE_THROWS(test_read.extract_sam_lines(false));
    }
    SECTION("Generating sam record for empty seq and qstring throws") {
        test_read.read_id = "test_read";
        REQUIRE_THROWS(test_read.extract_sam_lines(false));
    }
    SECTION("Generating sam record for mismatched seq and qstring throws") {
        test_read.read_id = "test_read";
        test_read.seq = "ACGTACGT";
        test_read.qstring = "!!!!";
        REQUIRE_THROWS(test_read.extract_sam_lines(false));
    }
    SECTION("Generating sam record for read with non-empty mappings throws") {
        test_read.read_id = "test_read";
        test_read.seq = "ACGTACGT";
        test_read.qstring = "!!!!!!!!";
        test_read.mappings.resize(1);
        REQUIRE_THROWS(test_read.extract_sam_lines(false));
    }
    SECTION("Generated sam record for unaligned read is correct") {
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

        auto lines = test_read.extract_sam_lines(false);
        REQUIRE(!lines.empty());
        auto& rec = lines[0];
        CHECK(rec->core.pos == -1);
        CHECK(rec->core.mpos == -1);
        CHECK(rec->core.l_qseq == 8);
        CHECK(rec->core.l_qname - rec->core.l_extranul - 1 ==
              9);  // qname length is stored with padded nulls to be 4-char aligned
        CHECK(rec->core.flag == 4);
        // Construct qstring from quality scores
        std::string qstring("");
        for (auto i = 0; i < rec->core.l_qseq; i++) {
            qstring += static_cast<char>(bam_get_qual(rec)[i] + 33);
        }
        CHECK(test_read.qstring == qstring);
        //Note; Tag generation is already tested in another test.
    }
}

void require_sam_tag_B_int_matches(const uint8_t* aux, const std::vector<int64_t>& expected) {
    int len = bam_auxB_len(aux);
    REQUIRE(len == expected.size());
    for (int i = 0; i < len; i++) {
        REQUIRE(expected[i] == bam_auxB2i(aux, i));
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
    read.read_id = "read";
    read.seq = "ACAGTGACTAAACTC";
    read.qstring = "***************";
    read.base_mod_probs = modbase_probs;
    read.is_duplex = false;

    std::string methylation_tag;
    SECTION("Methylation threshold is correctly applied") {
        read.base_mod_info = std::make_shared<dorado::utils::BaseModInfo>(modbase_alphabet,
                                                                          modbase_long_names, "");

        // Test generation
        const char* expected_methylation_tag_10_score = "A+a.,0,1;C+m.,1,0;";
        std::vector<int64_t> expected_methylation_tag_10_score_prob{20, 254, 252, 252};
        auto lines = read.extract_sam_lines(false, 10);
        REQUIRE(!lines.empty());
        bam1_t* aln = lines[0].get();
        CHECK_THAT(bam_aux2Z(bam_aux_get(aln, "MM")), Equals(expected_methylation_tag_10_score));
        require_sam_tag_B_int_matches(bam_aux_get(aln, "ML"),
                                      expected_methylation_tag_10_score_prob);

        // Test generation at higher rate excludes the correct mods.
        const char* expected_methylation_tag_50_score = "A+a.,2;C+m.,1,0;";
        std::vector<int64_t> expected_methylation_tag_50_score_prob{254, 252, 252};
        lines = read.extract_sam_lines(false, 50);
        REQUIRE(!lines.empty());
        aln = lines[0].get();
        CHECK_THAT(bam_aux2Z(bam_aux_get(aln, "MM")), Equals(expected_methylation_tag_50_score));
        require_sam_tag_B_int_matches(bam_aux_get(aln, "ML"),
                                      expected_methylation_tag_50_score_prob);

        // Test generation at max threshold rate excludes everything
        const char* expected_methylation_tag_255_score = "A+a.;C+m.;";
        std::vector<int64_t> expected_methylation_tag_255_score_prob{};
        lines = read.extract_sam_lines(false, 255);
        REQUIRE(!lines.empty());
        aln = lines[0].get();
        CHECK_THAT(bam_aux2Z(bam_aux_get(aln, "MM")), Equals(expected_methylation_tag_255_score));
        require_sam_tag_B_int_matches(bam_aux_get(aln, "ML"),
                                      expected_methylation_tag_255_score_prob);
    }

    SECTION("Test generation using CHEBI codes") {
        std::string modbase_long_names_CHEBI = "55555 12345";
        const char* expected_methylation_tag_CHEBI = "A+55555.,2;C+12345.,1,0;";
        std::vector<int64_t> expected_methylation_tag_CHEBI_prob{254, 252, 252};

        read.base_mod_info = std::make_shared<dorado::utils::BaseModInfo>(
                modbase_alphabet, modbase_long_names_CHEBI, "");
        auto lines = read.extract_sam_lines(false, 50);
        REQUIRE(!lines.empty());
        bam1_t* aln = lines[0].get();
        CHECK_THAT(bam_aux2Z(bam_aux_get(aln, "MM")), Equals(expected_methylation_tag_CHEBI));
        require_sam_tag_B_int_matches(bam_aux_get(aln, "ML"), expected_methylation_tag_CHEBI_prob);
    }

    SECTION("Test generation using AC context for A methylation") {
        std::string context = "XC:_:_:_";
        const char* expected_methylation_tag_with_context = "A+a?,0,1,2;C+m.,1,0;";
        std::vector<int64_t> expected_methylation_tag_with_context_prob{20, 254, 0, 252, 252};

        read.base_mod_info = std::make_shared<dorado::utils::BaseModInfo>(
                modbase_alphabet, modbase_long_names, context);

        auto lines = read.extract_sam_lines(false, 10);
        REQUIRE(!lines.empty());
        bam1_t* aln = lines[0].get();
        CHECK_THAT(bam_aux2Z(bam_aux_get(aln, "MM")),
                   Equals(expected_methylation_tag_with_context));
        require_sam_tag_B_int_matches(bam_aux_get(aln, "ML"),
                                      expected_methylation_tag_with_context_prob);
    }

    SECTION("Test handling of incorrect base names") {
        std::string modbase_long_names_unknown = "12mA 5mq";

        read.base_mod_info = std::make_shared<dorado::utils::BaseModInfo>(
                modbase_alphabet, modbase_long_names_unknown, "");
        auto lines = read.extract_sam_lines(false, 50);
        REQUIRE(!lines.empty());
        bam1_t* aln = lines[0].get();
        CHECK(bam_aux_get(aln, "MM") == NULL);
        CHECK(bam_aux_get(aln, "ML") == NULL);
    }
}

TEST_CASE(TEST_GROUP ": Test mean q-score generation", TEST_GROUP) {
    dorado::Read test_read;
    test_read.read_id = "read1";
    test_read.raw_data = torch::empty(4000);
    test_read.seq = "AAAAAAAAAA";
    test_read.qstring = "$$////////";
    test_read.sample_rate = 4000.0;
    test_read.shift = 128.3842f;
    test_read.scale = 8.258f;
    test_read.num_trimmed_samples = 132;
    test_read.attributes.mux = 2;
    test_read.attributes.read_number = 18501;
    test_read.attributes.channel_number = 5;
    test_read.attributes.start_time = "2017-04-29T09:10:04Z";
    test_read.attributes.fast5_filename = "batch_0.fast5";
    test_read.run_id = "xyz";
    test_read.model_name = "test_model";
    test_read.is_duplex = false;

    SECTION("Check with start pos = 0") {
        test_read.mean_qscore_start_pos = 0;
        CHECK(test_read.calculate_mean_qscore() == Approx(8.79143f));
    }

    SECTION("Check with start pos > 0") {
        test_read.mean_qscore_start_pos = 2;
        CHECK(test_read.calculate_mean_qscore() == Approx(14.9691f));
    }

    SECTION("Check start pos > qstring length returns 0.f") {
        test_read.mean_qscore_start_pos = 1000;
        CHECK(test_read.calculate_mean_qscore() == Approx(8.79143f));
    }

    SECTION("Check start pos = qstring length") {
        test_read.mean_qscore_start_pos = test_read.qstring.length();
        CHECK(test_read.calculate_mean_qscore() == Approx(8.79143f));
    }
}
