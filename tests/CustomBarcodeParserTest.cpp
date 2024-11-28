#include "TestUtils.h"
#include "demux/parse_custom_kit.h"
#include "demux/parse_custom_sequences.h"
#include "utils/barcode_kits.h"

#include <catch2/catch.hpp>

namespace fs = std::filesystem;

TEST_CASE("Parse custom single ended barcode arrangement", "[barcode_demux]") {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/custom_barcodes"));
    const auto test_file = data_dir / "test_kit_single_ended.toml";

    auto [kit_name, kit_info] = dorado::demux::parse_custom_arrangement(test_file.string());

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
    CHECK(!kit_info.rear_only_barcodes);
}

TEST_CASE("Parse custom single ended barcode arrangement with rear barcodes", "[barcode_demux]") {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/custom_barcodes"));
    const auto test_file = data_dir / "test_kit_single_ended_rear.toml";

    auto [kit_name, kit_info] = dorado::demux::parse_custom_arrangement(test_file.string());

    CHECK(kit_name == "test_kit_single_ended_rear_only");
    CHECK(kit_info.barcodes.size() == 4);

    CHECK(kit_info.name == "BC");
    CHECK(kit_info.top_front_flank == "C");
    CHECK(kit_info.top_rear_flank == "GTTTTCGCATTTATCGTGAAACGCTTTCGCGTTTTTCGTGCGCCGCTTCA");
    CHECK(kit_info.bottom_front_flank.empty());
    CHECK(kit_info.bottom_rear_flank.empty());
    CHECK(kit_info.barcodes2.empty());

    CHECK(!kit_info.double_ends);
    CHECK(!kit_info.ends_different);
    CHECK(kit_info.rear_only_barcodes);
}

TEST_CASE("Parse double ended barcode arrangement", "[barcode_demux]") {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/custom_barcodes"));
    const auto test_file = data_dir / "test_kit_double_ended.toml";

    auto [kit_name, kit_info] = dorado::demux::parse_custom_arrangement(test_file.string());

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

    auto [kit_name, kit_info] = dorado::demux::parse_custom_arrangement(test_file.string());

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
    CHECK_FALSE(kit_info.rear_only_barcodes);
}

TEST_CASE("Parse double ended barcode arrangement with different barcodes", "[barcode_demux]") {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/custom_barcodes"));
    const auto test_file = data_dir / "test_kit_ends_different_barcodes.toml";

    auto [kit_name, kit_info] = dorado::demux::parse_custom_arrangement(test_file.string());

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
    CHECK_FALSE(kit_info.rear_only_barcodes);
}

TEST_CASE("Parse kit with bad indices", "[barcode_demux]") {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/custom_barcodes"));
    const auto test_file = data_dir / "bad_double_ended_kit.toml";

    CHECK_THROWS(dorado::demux::parse_custom_arrangement(test_file.string()));
}

TEST_CASE("Parse kit with incomplete double ended settings", "[barcode_demux]") {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/custom_barcodes"));
    const auto test_file = data_dir / "bad_double_ended_kit_not_all_params_set.toml";

    CHECK_THROWS_WITH(dorado::demux::parse_custom_arrangement(test_file.string()),
                      Catch::Matchers::Contains(
                              "mask2_front mask2_rear and barcode2_pattern must all be set"));
}

TEST_CASE("Parse kit with no flanks", "[barcode_demux]") {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/custom_barcodes"));
    const auto test_file = data_dir / "flank_free_arrangement.toml";

    CHECK_THROWS_WITH(dorado::demux::parse_custom_arrangement(test_file.string()),
                      Catch::Matchers::Contains(
                              "At least one of mask1_front or mask1_rear needs to be specified"));
}

TEST_CASE("Parse custom barcode sequences", "[barcode_demux]") {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/custom_barcodes"));
    const auto test_sequences = data_dir / "test_sequences.fasta";

    auto barcode_sequences = dorado::demux::parse_custom_sequences(test_sequences.string());

    CHECK(barcode_sequences.size() == 4);
    CHECK(barcode_sequences["CUSTOM-BC01"] == "AAAAAA");
    CHECK(barcode_sequences["CUSTOM-BC02"] == "CCCCCC");
    CHECK(barcode_sequences["CUSTOM-BC03"] == "TTTTTT");
    CHECK(barcode_sequences["CUSTOM-BC04"] == "GGGGGG");
}

TEST_CASE("Parse custom barcode scoring params", "[barcode_demux]") {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/custom_barcodes"));
    const auto test_params_file = data_dir / "scoring_params.toml";

    dorado::barcode_kits::BarcodeKitScoringParams default_params;
    auto scoring_params =
            dorado::demux::parse_scoring_params(test_params_file.string(), default_params);

    CHECK(scoring_params.max_barcode_penalty == 10);
    CHECK(scoring_params.barcode_end_proximity == 75);
    CHECK(scoring_params.min_barcode_penalty_dist == 3);
    CHECK(scoring_params.min_separation_only_dist == 5);
    CHECK(scoring_params.flank_left_pad == 5);
    CHECK(scoring_params.flank_right_pad == 10);
    CHECK(scoring_params.front_barcode_window == 150);
    CHECK(scoring_params.rear_barcode_window == 150);
    CHECK(scoring_params.min_flank_score == Approx(0.5f));
}

TEST_CASE("Parse default scoring params", "[barcode_demux]") {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/custom_barcodes"));
    const auto test_params_file = data_dir / "test_kit_single_ended.toml";

    dorado::barcode_kits::BarcodeKitScoringParams default_params;
    auto scoring_params =
            dorado::demux::parse_scoring_params(test_params_file.string(), default_params);

    CHECK(scoring_params.max_barcode_penalty == default_params.max_barcode_penalty);
    CHECK(scoring_params.barcode_end_proximity == default_params.barcode_end_proximity);
    CHECK(scoring_params.min_barcode_penalty_dist == default_params.min_barcode_penalty_dist);
    CHECK(scoring_params.min_separation_only_dist == default_params.min_separation_only_dist);
    CHECK(scoring_params.flank_left_pad == default_params.flank_left_pad);
    CHECK(scoring_params.flank_right_pad == default_params.flank_right_pad);
}

TEST_CASE("Check for normalized id pattern", "[barcode_demux]") {
    CHECK(dorado::demux::check_normalized_id_pattern("BC%02i"));
    CHECK(dorado::demux::check_normalized_id_pattern("abcd%25i"));
    CHECK(dorado::demux::check_normalized_id_pattern("%2i"));

    CHECK_FALSE(dorado::demux::check_normalized_id_pattern("ab"));
    CHECK_FALSE(dorado::demux::check_normalized_id_pattern("ab%"));
    CHECK_FALSE(dorado::demux::check_normalized_id_pattern("ab%d"));
    CHECK_FALSE(dorado::demux::check_normalized_id_pattern("ab%02"));
    CHECK_FALSE(dorado::demux::check_normalized_id_pattern("ab%02f"));
    CHECK_FALSE(dorado::demux::check_normalized_id_pattern("ab%02iab"));
}
