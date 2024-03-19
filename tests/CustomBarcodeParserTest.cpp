#include "TestUtils.h"
#include "demux/parse_custom_sequences.h"
#include "utils/parse_custom_kit.h"

#include <catch2/catch.hpp>

namespace fs = std::filesystem;

TEST_CASE("Parse custom single ended barcode arrangement", "[barcode_demux]") {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/custom_barcodes"));
    const auto test_file = data_dir / "test_kit_single_ended.toml";

    auto [kit_name, kit_info] = *dorado::barcode_kits::parse_custom_arrangement(test_file.string());

    CHECK(kit_name == "test_kit_single_ended");
    CHECK(kit_info.barcodes.size() == 4);

    CHECK(kit_info.name == "BC");
    CHECK(kit_info.top_front_flank == "C");
    CHECK(kit_info.top_rear_flank == "GTTTTCGCATTTATCGTGAAACGCTTTCGCGTTTTTCGTGCGCCGCTTCA");
    CHECK(kit_info.bottom_front_flank.empty());
    CHECK(kit_info.bottom_rear_flank.empty());
    CHECK(kit_info.barcodes2.empty());

    CHECK(!kit_info.double_ends);
    CHECK(!kit_info.ends_different);
}

TEST_CASE("Parse double ended barcode arrangement", "[barcode_demux]") {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/custom_barcodes"));
    const auto test_file = data_dir / "test_kit_double_ended.toml";

    auto [kit_name, kit_info] = *dorado::barcode_kits::parse_custom_arrangement(test_file.string());

    CHECK(kit_name == "test_kit_double_ended");
    CHECK(kit_info.barcodes.size() == 24);
    CHECK(kit_info.barcodes2.size() == 24);

    CHECK(kit_info.name == "BC");
    CHECK(kit_info.top_front_flank == "CCCC");
    CHECK(kit_info.top_rear_flank == "GTTTTCG");
    CHECK(kit_info.bottom_front_flank == "CCCC");
    CHECK(kit_info.bottom_rear_flank == "GTTTTCG");

    CHECK(kit_info.double_ends);
    CHECK(!kit_info.ends_different);
}

TEST_CASE("Parse double ended barcode arrangement with different flanks", "[barcode_demux]") {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/custom_barcodes"));
    const auto test_file = data_dir / "test_kit_ends_different_flanks.toml";

    auto [kit_name, kit_info] = *dorado::barcode_kits::parse_custom_arrangement(test_file.string());

    CHECK(kit_name == "test_kit_ends_different_flanks");
    CHECK(kit_info.barcodes.size() == 96);
    CHECK(kit_info.barcodes2.size() == 96);

    CHECK(kit_info.name == "NB");
    CHECK(kit_info.top_front_flank == "AAAA");
    CHECK(kit_info.top_rear_flank == "TTTTT");
    CHECK(kit_info.bottom_front_flank == "CCCC");
    CHECK(kit_info.bottom_rear_flank == "GGGG");

    CHECK(kit_info.double_ends);
    CHECK(kit_info.ends_different);
}

TEST_CASE("Parse double ended barcode arrangement with different barcodes", "[barcode_demux]") {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/custom_barcodes"));
    const auto test_file = data_dir / "test_kit_ends_different_barcodes.toml";

    auto [kit_name, kit_info] = *dorado::barcode_kits::parse_custom_arrangement(test_file.string());

    CHECK(kit_name == "test_kit_ends_different_barcodes");
    CHECK(kit_info.barcodes.size() == 12);
    CHECK(kit_info.barcodes2.size() == 12);

    CHECK(kit_info.name == "BC");
    CHECK(kit_info.top_front_flank == "C");
    CHECK(kit_info.top_rear_flank == "G");
    CHECK(kit_info.bottom_front_flank == "C");
    CHECK(kit_info.bottom_rear_flank == "G");

    CHECK(kit_info.double_ends);
    CHECK(kit_info.ends_different);
}

TEST_CASE("Parse kit with bad indices", "[barcode_demux]") {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/custom_barcodes"));
    const auto test_file = data_dir / "bad_double_ended_kit.toml";

    CHECK_THROWS(dorado::barcode_kits::parse_custom_arrangement(test_file.string()));
}

TEST_CASE("Parse kit with incomplete double ended settings", "[barcode_demux]") {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/custom_barcodes"));
    const auto test_file = data_dir / "bad_double_ended_kit_not_all_params_set.toml";

    CHECK_THROWS_WITH(dorado::barcode_kits::parse_custom_arrangement(test_file.string()),
                      Catch::Matchers::Contains(
                              "mask2_front mask2_rear and barcode2_pattern must all be set"));
}

TEST_CASE("Parse custom barcode sequences", "[barcode_demux]") {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/custom_barcodes"));
    const auto test_sequences = data_dir / "test_sequences.fasta";

    auto barcode_sequences = dorado::demux::parse_custom_sequences(test_sequences.string());

    CHECK(barcode_sequences.size() == 4);
    CHECK(barcode_sequences["BC01"] == "AAAAAA");
    CHECK(barcode_sequences["BC02"] == "CCCCCC");
    CHECK(barcode_sequences["BC03"] == "TTTTTT");
    CHECK(barcode_sequences["BC04"] == "GGGGGG");
}

TEST_CASE("Parse custom barcode scoring params", "[barcode_demux]") {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/custom_barcodes"));
    const auto test_params_file = data_dir / "scoring_params.toml";

    auto scoring_params = dorado::barcode_kits::parse_scoring_params(test_params_file.string());

    CHECK(scoring_params.min_soft_barcode_threshold == Approx(0.1f));
    CHECK(scoring_params.min_hard_barcode_threshold == Approx(0.2f));
    CHECK(scoring_params.min_soft_flank_threshold == Approx(0.3f));
    CHECK(scoring_params.min_hard_flank_threshold == Approx(0.4f));
    CHECK(scoring_params.min_barcode_score_dist == Approx(0.5));
}

TEST_CASE("Parse default scoring params", "[barcode_demux]") {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/custom_barcodes"));
    const auto test_params_file = data_dir / "test_kit_single_ended.toml";

    dorado::barcode_kits::BarcodeKitScoringParams default_params;

    auto scoring_params = dorado::barcode_kits::parse_scoring_params(test_params_file.string());

    CHECK(scoring_params.min_soft_barcode_threshold ==
          Approx(default_params.min_soft_barcode_threshold));
    CHECK(scoring_params.min_hard_barcode_threshold ==
          Approx(default_params.min_hard_barcode_threshold));
    CHECK(scoring_params.min_soft_flank_threshold ==
          Approx(default_params.min_soft_flank_threshold));
    CHECK(scoring_params.min_hard_flank_threshold ==
          Approx(default_params.min_hard_flank_threshold));
    CHECK(scoring_params.min_barcode_score_dist == Approx(default_params.min_barcode_score_dist));
}

TEST_CASE("Check for normalized id pattern", "[barcode_demux]") {
    CHECK(dorado::barcode_kits::check_normalized_id_pattern("BC%02i"));
    CHECK(dorado::barcode_kits::check_normalized_id_pattern("abcd%25i"));
    CHECK(dorado::barcode_kits::check_normalized_id_pattern("%2i"));

    CHECK_FALSE(dorado::barcode_kits::check_normalized_id_pattern("ab"));
    CHECK_FALSE(dorado::barcode_kits::check_normalized_id_pattern("ab%"));
    CHECK_FALSE(dorado::barcode_kits::check_normalized_id_pattern("ab%d"));
    CHECK_FALSE(dorado::barcode_kits::check_normalized_id_pattern("ab%02"));
    CHECK_FALSE(dorado::barcode_kits::check_normalized_id_pattern("ab%02f"));
    CHECK_FALSE(dorado::barcode_kits::check_normalized_id_pattern("ab%02iab"));
}
