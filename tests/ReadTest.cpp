#include "read_pipeline/ReadPipeline.h"
#include "utils/types.h"

#include <ATen/Functions.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <htslib/sam.h>

#define TEST_GROUP "[ReadTest]"

using Catch::Matchers::Equals;

CATCH_TEST_CASE(TEST_GROUP ": Test tag generation", TEST_GROUP) {
    dorado::ReadCommon read_common;
    read_common.read_id = "read1";
    read_common.raw_data = at::empty(4000);
    read_common.seq = "ACGT";
    read_common.qstring = "////";
    read_common.sample_rate = 4000;
    read_common.shift = 128.3842f;
    read_common.scale = 8.258f;
    read_common.scaling_method = "quantile";
    read_common.num_trimmed_samples = 132;
    read_common.attributes.mux = 2;
    read_common.attributes.read_number = 18501;
    read_common.attributes.channel_number = 5;
    read_common.attributes.start_time = "2017-04-29T09:10:04Z";
    read_common.attributes.filename = "batch_0.fast5";
    read_common.run_id = "xyz";
    read_common.model_name = "test_model";
    read_common.is_duplex = false;
    read_common.parent_read_id = "parent_read";
    read_common.split_point = 0;

    CATCH_SECTION("Basic") {
        auto alignments = read_common.extract_sam_lines(false, std::nullopt, false);
        CATCH_REQUIRE(alignments.size() == 1);
        bam1_t* aln = alignments[0].get();

        CATCH_CHECK(bam_aux2f(bam_aux_get(aln, "qs")) == 14.0f);
        CATCH_CHECK(bam_aux2i(bam_aux_get(aln, "ns")) == 4132);
        CATCH_CHECK(bam_aux2i(bam_aux_get(aln, "ts")) == 132);
        CATCH_CHECK(bam_aux2i(bam_aux_get(aln, "mx")) == 2);
        CATCH_CHECK(bam_aux2i(bam_aux_get(aln, "ch")) == 5);
        CATCH_CHECK(bam_aux2i(bam_aux_get(aln, "rn")) == 18501);
        CATCH_CHECK(bam_aux2i(bam_aux_get(aln, "rn")) == 18501);
        CATCH_CHECK(bam_aux2i(bam_aux_get(aln, "dx")) == 0);
        CATCH_CHECK(bam_aux2i(bam_aux_get(aln, "sp")) == 0);
        CATCH_CHECK(bam_aux_get(aln, "pt") == nullptr);

        CATCH_CHECK(bam_aux2f(bam_aux_get(aln, "du")) == Catch::Approx(1.033).margin(1e-6));
        CATCH_CHECK(bam_aux2f(bam_aux_get(aln, "sm")) == 128.3842f);
        CATCH_CHECK(bam_aux2f(bam_aux_get(aln, "sd")) == 8.258f);

        CATCH_CHECK_THAT(bam_aux2Z(bam_aux_get(aln, "st")), Equals("2017-04-29T09:10:04Z"));
        CATCH_CHECK_THAT(bam_aux2Z(bam_aux_get(aln, "fn")), Equals("batch_0.fast5"));
        CATCH_CHECK_THAT(bam_aux2Z(bam_aux_get(aln, "sv")), Equals("quantile"));
        CATCH_CHECK_THAT(bam_aux2Z(bam_aux_get(aln, "RG")), Equals("xyz_test_model"));
        CATCH_CHECK_THAT(bam_aux2Z(bam_aux_get(aln, "pi")), Equals("parent_read"));

        CATCH_CHECK(bam_aux_get(aln, "BC") == nullptr);
    }

    CATCH_SECTION("Duplex") {
        // Update read to be duplex
        auto was_duplex = std::exchange(read_common.is_duplex, true);

        auto alignments = read_common.extract_sam_lines(false, std::nullopt, false);
        CATCH_REQUIRE(alignments.size() == 1);
        auto* aln = alignments[0].get();

        CATCH_CHECK(bam_aux2i(bam_aux_get(aln, "dx")) == 1);
        CATCH_CHECK_THAT(bam_aux2Z(bam_aux_get(aln, "RG")), Equals("xyz_test_model"));
        CATCH_CHECK_THAT(bam_aux2Z(bam_aux_get(aln, "pi")), Equals("parent_read"));

        read_common.is_duplex = was_duplex;
    }

    CATCH_SECTION("Duplex Parent") {
        // Update read to be duplex parent
        auto alignments = read_common.extract_sam_lines(false, std::nullopt, true);
        CATCH_REQUIRE(alignments.size() == 1);
        auto* aln = alignments[0].get();

        CATCH_CHECK(bam_aux2i(bam_aux_get(aln, "dx")) == -1);
    }

    CATCH_SECTION("No model") {
        auto old_model = std::exchange(read_common.model_name, "");

        auto alignments = read_common.extract_sam_lines(false, std::nullopt, false);
        CATCH_REQUIRE(alignments.size() == 1);
        auto* aln = alignments[0].get();

        CATCH_CHECK_THAT(bam_aux2Z(bam_aux_get(aln, "RG")), Equals("xyz_unknown"));

        read_common.model_name = old_model;
    }

    CATCH_SECTION("No model or run_id") {
        auto old_model = std::exchange(read_common.model_name, "");
        auto old_run_id = std::exchange(read_common.run_id, "");

        auto alignments = read_common.extract_sam_lines(false, std::nullopt, false);
        CATCH_REQUIRE(alignments.size() == 1);
        auto* aln = alignments[0].get();

        CATCH_CHECK(bam_aux_get(aln, "RG") == nullptr);

        read_common.model_name = old_model;
        read_common.run_id = old_run_id;
    }

    CATCH_SECTION("Barcode") {
        auto old_barcode = std::exchange(read_common.barcode, "kit_barcode02");

        auto alignments = read_common.extract_sam_lines(false, std::nullopt, false);
        CATCH_REQUIRE(alignments.size() == 1);
        auto* aln = alignments[0].get();

        CATCH_CHECK_THAT(bam_aux2Z(bam_aux_get(aln, "RG")), Equals("xyz_test_model_kit_barcode02"));
        CATCH_CHECK_THAT(bam_aux2Z(bam_aux_get(aln, "BC")), Equals("kit_barcode02"));

        read_common.barcode = old_barcode;
    }

    CATCH_SECTION("Barcode unclassified") {
        auto old_barcode = std::exchange(read_common.barcode, dorado::UNCLASSIFIED);

        auto alignments = read_common.extract_sam_lines(false, std::nullopt, false);
        CATCH_REQUIRE(alignments.size() == 1);
        auto* aln = alignments[0].get();

        CATCH_CHECK_THAT(bam_aux2Z(bam_aux_get(aln, "RG")), Equals("xyz_test_model"));
        CATCH_CHECK(bam_aux_get(aln, "BC") == nullptr);

        read_common.barcode = old_barcode;
    }

    CATCH_SECTION("PolyA tail length") {
        auto old_tail_length = std::exchange(read_common.rna_poly_tail_length,
                                             dorado::ReadCommon::POLY_TAIL_NOT_ENABLED);
        auto alignments = read_common.extract_sam_lines(false, std::nullopt, false);
        CATCH_REQUIRE(alignments.size() == 1);
        auto* aln = alignments[0].get();
        CATCH_CHECK(bam_aux_get(aln, "pt") == nullptr);

        read_common.rna_poly_tail_length = dorado::ReadCommon::POLY_TAIL_NO_ANCHOR_FOUND;
        alignments = read_common.extract_sam_lines(false, std::nullopt, false);
        CATCH_REQUIRE(alignments.size() == 1);
        aln = alignments[0].get();
        CATCH_CHECK(bam_aux2i(bam_aux_get(aln, "pt")) ==
                    dorado::ReadCommon::POLY_TAIL_NO_ANCHOR_FOUND);

        read_common.rna_poly_tail_length = 20;
        alignments = read_common.extract_sam_lines(false, std::nullopt, false);
        CATCH_REQUIRE(alignments.size() == 1);
        aln = alignments[0].get();
        CATCH_CHECK(bam_aux2i(bam_aux_get(aln, "pt")) == 20);

        read_common.rna_poly_tail_length = old_tail_length;
    }
}

CATCH_TEST_CASE(TEST_GROUP ": Test sam record generation", TEST_GROUP) {
    dorado::SimplexRead test_read{};
    CATCH_SECTION("Generating sam record for empty read throws") {
        CATCH_REQUIRE_THROWS(test_read.read_common.extract_sam_lines(false, std::nullopt, false));
    }
    CATCH_SECTION("Generating sam record for empty seq and qstring throws") {
        test_read.read_common.read_id = "test_read";
        CATCH_REQUIRE_THROWS(test_read.read_common.extract_sam_lines(false, std::nullopt, false));
    }
    CATCH_SECTION("Generating sam record for mismatched seq and qstring throws") {
        test_read.read_common.read_id = "test_read";
        test_read.read_common.seq = "ACGTACGT";
        test_read.read_common.qstring = "!!!!";
        CATCH_REQUIRE_THROWS(test_read.read_common.extract_sam_lines(false, std::nullopt, false));
    }

    CATCH_SECTION("Generated sam record for unaligned read is correct") {
        test_read.read_common.raw_data = at::empty(4000);
        test_read.read_common.sample_rate = 4000;
        test_read.read_common.shift = 128.3842f;
        test_read.read_common.scale = 8.258f;
        test_read.read_common.read_id = "test_read";
        test_read.read_common.seq = "ACGTACGT";
        test_read.read_common.qstring = "********";
        test_read.read_common.num_trimmed_samples = 132;
        test_read.read_common.attributes.mux = 2;
        test_read.read_common.attributes.read_number = 18501;
        test_read.read_common.attributes.channel_number = 5;
        test_read.read_common.attributes.start_time = "2017-04-29T09:10:04Z";
        test_read.read_common.attributes.filename = "batch_0.fast5";

        auto lines = test_read.read_common.extract_sam_lines(false, std::nullopt, false);
        CATCH_REQUIRE(!lines.empty());
        auto& rec = lines[0];
        CATCH_CHECK(rec->core.pos == -1);
        CATCH_CHECK(rec->core.mpos == -1);
        CATCH_CHECK(rec->core.l_qseq == 8);
        CATCH_CHECK(rec->core.l_qname - rec->core.l_extranul - 1 ==
                    9);  // qname length is stored with padded nulls to be 4-char aligned
        CATCH_CHECK(rec->core.flag == 4);
        // Construct qstring from quality scores
        std::string qstring("");
        for (auto i = 0; i < rec->core.l_qseq; i++) {
            qstring += static_cast<char>(bam_get_qual(rec)[i] + 33);
        }
        CATCH_CHECK(test_read.read_common.qstring == qstring);
        //Note; Tag generation is already tested in another test.
    }
}

namespace {

void require_sam_tag_B_int_matches(const uint8_t* aux, const std::vector<int64_t>& expected) {
    int len = bam_auxB_len(aux);
    CATCH_REQUIRE(size_t(len) == expected.size());
    for (int i = 0; i < len; i++) {
        CATCH_REQUIRE(expected[i] == bam_auxB2i(aux, i));
    }
}

}  // namespace

CATCH_TEST_CASE(TEST_GROUP ": Methylation tag generation", TEST_GROUP) {
    std::vector<std::string> modbase_alphabet = {"A", "a", "C", "m", "G", "T"};
    std::string modbase_long_names = "6mA 5mC";
    std::vector<uint8_t> modbase_probs = {
            235, 20,  0,   0,   0,   0,    // A 6mA (weak call)
            0,   0,   255, 0,   0,   0,    // C
            255, 0,   0,   0,   0,   0,    // A
            0,   0,   0,   0,   255, 0,    // G
            0,   0,   0,   0,   0,   255,  // T
            0,   0,   0,   0,   255, 0,    // G
            1,   254, 0,   0,   0,   0,    // A 6mA
            0,   0,   3,   252, 0,   0,    // C 5mC
            0,   0,   0,   0,   0,   255,  // T
            255, 0,   0,   0,   0,   0,    // A
            255, 0,   0,   0,   0,   0,    // A
            255, 0,   0,   0,   0,   0,    // A
            0,   0,   3,   252, 0,   0,    // C 5mC
            0,   0,   0,   0,   0,   255,  // T
            0,   0,   255, 0,   0,   0,    // C
    };

    dorado::ReadCommon read_common;
    read_common.read_id = "read";
    read_common.seq = "ACAGTGACTAAACTC";
    read_common.qstring = "***************";
    read_common.base_mod_probs = modbase_probs;
    read_common.is_duplex = false;

    std::string methylation_tag;
    CATCH_SECTION("Methylation threshold is correctly applied") {
        read_common.mod_base_info =
                std::make_shared<dorado::ModBaseInfo>(modbase_alphabet, modbase_long_names, "");

        // Test generation
        const char* expected_methylation_tag_10_score = "A+a.,0,1;C+m.,1,0;";
        std::vector<int64_t> expected_methylation_tag_10_score_prob{20, 254, 252, 252};
        auto lines = read_common.extract_sam_lines(false, static_cast<uint8_t>(10), false);
        CATCH_REQUIRE(!lines.empty());
        bam1_t* aln = lines[0].get();
        CATCH_CHECK_THAT(bam_aux2Z(bam_aux_get(aln, "MM")),
                         Equals(expected_methylation_tag_10_score));
        require_sam_tag_B_int_matches(bam_aux_get(aln, "ML"),
                                      expected_methylation_tag_10_score_prob);

        // Test generation at higher rate excludes the correct mods.
        const char* expected_methylation_tag_50_score = "A+a.,2;C+m.,1,0;";
        std::vector<int64_t> expected_methylation_tag_50_score_prob{254, 252, 252};
        lines = read_common.extract_sam_lines(false, static_cast<uint8_t>(50), false);
        CATCH_REQUIRE(!lines.empty());
        aln = lines[0].get();
        CATCH_CHECK_THAT(bam_aux2Z(bam_aux_get(aln, "MM")),
                         Equals(expected_methylation_tag_50_score));
        require_sam_tag_B_int_matches(bam_aux_get(aln, "ML"),
                                      expected_methylation_tag_50_score_prob);

        // Test generation at max threshold rate excludes everything
        const char* expected_methylation_tag_255_score = "A+a.;C+m.;";
        std::vector<int64_t> expected_methylation_tag_255_score_prob{};
        lines = read_common.extract_sam_lines(false, static_cast<uint8_t>(255), false);
        CATCH_REQUIRE(!lines.empty());
        aln = lines[0].get();
        CATCH_CHECK_THAT(bam_aux2Z(bam_aux_get(aln, "MM")),
                         Equals(expected_methylation_tag_255_score));
        require_sam_tag_B_int_matches(bam_aux_get(aln, "ML"),
                                      expected_methylation_tag_255_score_prob);
    }

    CATCH_SECTION("Test generation using CHEBI codes") {
        auto modbase_alphabet_CHEBI = modbase_alphabet;
        modbase_alphabet_CHEBI[1] = "55555";
        modbase_alphabet_CHEBI[3] = "12345";
        const char* expected_methylation_tag_CHEBI = "A+55555.,2;C+12345.,1,0;";
        std::vector<int64_t> expected_methylation_tag_CHEBI_prob{254, 252, 252};

        read_common.mod_base_info = std::make_shared<dorado::ModBaseInfo>(modbase_alphabet_CHEBI,
                                                                          modbase_long_names, "");
        auto lines = read_common.extract_sam_lines(false, static_cast<uint8_t>(50), false);
        CATCH_REQUIRE(!lines.empty());
        bam1_t* aln = lines[0].get();
        CATCH_CHECK_THAT(bam_aux2Z(bam_aux_get(aln, "MM")), Equals(expected_methylation_tag_CHEBI));
        require_sam_tag_B_int_matches(bam_aux_get(aln, "ML"), expected_methylation_tag_CHEBI_prob);
    }

    CATCH_SECTION("Test generation using AC context for A methylation") {
        std::string context = "XC:_:_:_";
        const char* expected_methylation_tag_with_context = "A+a?,0,1,2;C+m.,1,0;";
        std::vector<int64_t> expected_methylation_tag_with_context_prob{20, 254, 0, 252, 252};

        read_common.mod_base_info = std::make_shared<dorado::ModBaseInfo>(
                modbase_alphabet, modbase_long_names, context);

        auto lines = read_common.extract_sam_lines(false, static_cast<uint8_t>(10), false);
        CATCH_REQUIRE(!lines.empty());
        bam1_t* aln = lines[0].get();
        CATCH_CHECK_THAT(bam_aux2Z(bam_aux_get(aln, "MM")),
                         Equals(expected_methylation_tag_with_context));
        require_sam_tag_B_int_matches(bam_aux_get(aln, "ML"),
                                      expected_methylation_tag_with_context_prob);
    }

    CATCH_SECTION("Test generation using DRACH context for A methylation") {
        std::string context = "DRXCH:_:_:_";
        const char* expected_methylation_tag_with_context = "A+a?,2,2;C+m.,1,0;";
        std::vector<int64_t> expected_methylation_tag_with_context_prob{254, 0, 252, 252};

        read_common.mod_base_info = std::make_shared<dorado::ModBaseInfo>(
                modbase_alphabet, modbase_long_names, context);

        auto lines = read_common.extract_sam_lines(false, static_cast<uint8_t>(10), false);
        CATCH_REQUIRE(!lines.empty());
        bam1_t* aln = lines[0].get();
        CATCH_CHECK_THAT(bam_aux2Z(bam_aux_get(aln, "MM")),
                         Equals(expected_methylation_tag_with_context));
        require_sam_tag_B_int_matches(bam_aux_get(aln, "ML"),
                                      expected_methylation_tag_with_context_prob);
    }

    CATCH_SECTION("Test handling of incorrect base names") {
        auto modbase_alphabet_unknown = modbase_alphabet;
        modbase_alphabet_unknown[1] = "12mA";
        modbase_alphabet_unknown[3] = "mq";

        read_common.mod_base_info = std::make_shared<dorado::ModBaseInfo>(modbase_alphabet_unknown,
                                                                          modbase_long_names, "");
        auto lines = read_common.extract_sam_lines(false, static_cast<uint8_t>(50), false);
        CATCH_REQUIRE(!lines.empty());
        bam1_t* aln = lines[0].get();
        CATCH_CHECK(bam_aux_get(aln, "MM") == NULL);
        CATCH_CHECK(bam_aux_get(aln, "ML") == NULL);
    }
}

CATCH_TEST_CASE(TEST_GROUP ": Test mean q-score generation", TEST_GROUP) {
    dorado::ReadCommon read_common;
    read_common.read_id = "read1";
    read_common.raw_data = at::empty(4000);
    read_common.seq = "AAAAAAAAAA";
    read_common.qstring = "$$////////";
    read_common.sample_rate = 4000;
    read_common.shift = 128.3842f;
    read_common.scale = 8.258f;
    read_common.num_trimmed_samples = 132;
    read_common.attributes.mux = 2;
    read_common.attributes.read_number = 18501;
    read_common.attributes.channel_number = 5;
    read_common.attributes.start_time = "2017-04-29T09:10:04Z";
    read_common.attributes.filename = "batch_0.fast5";
    read_common.run_id = "xyz";
    read_common.model_name = "test_model";
    read_common.is_duplex = false;

    CATCH_SECTION("Check with start pos = 0") {
        read_common.mean_qscore_start_pos = 0;
        CATCH_CHECK(read_common.calculate_mean_qscore() == Catch::Approx(8.79143f));
    }

    CATCH_SECTION("Check with start pos > 0") {
        read_common.mean_qscore_start_pos = 2;
        CATCH_CHECK(read_common.calculate_mean_qscore() == Catch::Approx(14.0f));
    }

    CATCH_SECTION("Check start pos > qstring length returns 0.f") {
        read_common.mean_qscore_start_pos = 1000;
        CATCH_CHECK(read_common.calculate_mean_qscore() == Catch::Approx(8.79143f));
    }

    CATCH_SECTION("Check start pos = qstring length") {
        read_common.mean_qscore_start_pos = int(read_common.qstring.length());
        CATCH_CHECK(read_common.calculate_mean_qscore() == Catch::Approx(8.79143f));
    }
}
