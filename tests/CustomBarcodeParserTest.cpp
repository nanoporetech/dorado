#include "TestUtils.h"
#include "demux/parse_custom_kit.h"
#include "demux/parse_custom_sequences.h"
#include "utils/barcode_kits.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

namespace fs = std::filesystem;

CATCH_TEST_CASE("Parse custom single ended barcode arrangement", "[barcode_demux]") {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/custom_barcodes"));
    const auto test_file = data_dir / "test_kit_single_ended.toml";

    auto [kit_name, kit_info] = dorado::demux::parse_custom_arrangement(test_file.string());

    CATCH_CHECK(kit_name == "test_kit_single_ended");
    CATCH_CHECK(kit_info.barcodes.size() == 4);

    CATCH_CHECK(kit_info.name == "BC");
    CATCH_CHECK(kit_info.top_front_flank == "C");
    CATCH_CHECK(kit_info.top_rear_flank == "GTTTTCGCATTTATCGTGAAACGCTTTCGCGTTTTTCGTGCGCCGCTTCA");
    CATCH_CHECK(kit_info.bottom_front_flank.empty());
    CATCH_CHECK(kit_info.bottom_rear_flank.empty());
    CATCH_CHECK(kit_info.barcodes2.empty());

    CATCH_CHECK(!kit_info.double_ends);
    CATCH_CHECK(!kit_info.ends_different);
    CATCH_CHECK(!kit_info.rear_only_barcodes);
}

CATCH_TEST_CASE("Parse custom single ended barcode arrangement with rear barcodes",
                "[barcode_demux]") {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/custom_barcodes"));
    const auto test_file = data_dir / "test_kit_single_ended_rear.toml";

    auto [kit_name, kit_info] = dorado::demux::parse_custom_arrangement(test_file.string());

    CATCH_CHECK(kit_name == "test_kit_single_ended_rear_only");
    CATCH_CHECK(kit_info.barcodes.size() == 4);

    CATCH_CHECK(kit_info.name == "BC");
    CATCH_CHECK(kit_info.top_front_flank == "C");
    CATCH_CHECK(kit_info.top_rear_flank == "GTTTTCGCATTTATCGTGAAACGCTTTCGCGTTTTTCGTGCGCCGCTTCA");
    CATCH_CHECK(kit_info.bottom_front_flank.empty());
    CATCH_CHECK(kit_info.bottom_rear_flank.empty());
    CATCH_CHECK(kit_info.barcodes2.empty());

    CATCH_CHECK(!kit_info.double_ends);
    CATCH_CHECK(!kit_info.ends_different);
    CATCH_CHECK(kit_info.rear_only_barcodes);
}

CATCH_TEST_CASE("Parse double ended barcode arrangement", "[barcode_demux]") {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/custom_barcodes"));
    const auto test_file = data_dir / "test_kit_double_ended.toml";

    auto [kit_name, kit_info] = dorado::demux::parse_custom_arrangement(test_file.string());

    CATCH_CHECK(kit_name == "test_kit_double_ended");
    CATCH_CHECK(kit_info.barcodes.size() == 24);
    CATCH_CHECK(kit_info.barcodes2.size() == 24);

    CATCH_CHECK(kit_info.name == "BC");
    CATCH_CHECK(kit_info.top_front_flank == "CCCC");
    CATCH_CHECK(kit_info.top_rear_flank == "GTTTTCG");
    CATCH_CHECK(kit_info.bottom_front_flank == "CCCC");
    CATCH_CHECK(kit_info.bottom_rear_flank == "GTTTTCG");

    CATCH_CHECK(kit_info.double_ends);
    CATCH_CHECK(!kit_info.ends_different);
}

CATCH_TEST_CASE("Parse double ended barcode arrangement with different flanks", "[barcode_demux]") {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/custom_barcodes"));
    const auto test_file = data_dir / "test_kit_ends_different_flanks.toml";

    auto [kit_name, kit_info] = dorado::demux::parse_custom_arrangement(test_file.string());

    CATCH_CHECK(kit_name == "test_kit_ends_different_flanks");
    CATCH_CHECK(kit_info.barcodes.size() == 96);
    CATCH_CHECK(kit_info.barcodes2.size() == 96);

    CATCH_CHECK(kit_info.name == "NB");
    CATCH_CHECK(kit_info.top_front_flank == "AAAA");
    CATCH_CHECK(kit_info.top_rear_flank == "TTTTT");
    CATCH_CHECK(kit_info.bottom_front_flank == "CCCC");
    CATCH_CHECK(kit_info.bottom_rear_flank == "GGGG");

    CATCH_CHECK(kit_info.double_ends);
    CATCH_CHECK(kit_info.ends_different);
    CATCH_CHECK_FALSE(kit_info.rear_only_barcodes);
}

CATCH_TEST_CASE("Parse double ended barcode arrangement with different barcodes",
                "[barcode_demux]") {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/custom_barcodes"));
    const auto test_file = data_dir / "test_kit_ends_different_barcodes.toml";

    auto [kit_name, kit_info] = dorado::demux::parse_custom_arrangement(test_file.string());

    CATCH_CHECK(kit_name == "test_kit_ends_different_barcodes");
    CATCH_CHECK(kit_info.barcodes.size() == 12);
    CATCH_CHECK(kit_info.barcodes2.size() == 12);

    CATCH_CHECK(kit_info.name == "BC");
    CATCH_CHECK(kit_info.top_front_flank == "C");
    CATCH_CHECK(kit_info.top_rear_flank == "G");
    CATCH_CHECK(kit_info.bottom_front_flank == "C");
    CATCH_CHECK(kit_info.bottom_rear_flank == "G");

    CATCH_CHECK(kit_info.double_ends);
    CATCH_CHECK(kit_info.ends_different);
    CATCH_CHECK_FALSE(kit_info.rear_only_barcodes);
}

CATCH_TEST_CASE("Parse kit with bad indices", "[barcode_demux]") {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/custom_barcodes"));
    const auto test_file = data_dir / "bad_double_ended_kit.toml";

    CATCH_CHECK_THROWS(dorado::demux::parse_custom_arrangement(test_file.string()));
}

CATCH_TEST_CASE("Parse kit with incomplete double ended settings", "[barcode_demux]") {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/custom_barcodes"));
    const auto test_file = data_dir / "bad_double_ended_kit_not_all_params_set.toml";

    CATCH_CHECK_THROWS_WITH(dorado::demux::parse_custom_arrangement(test_file.string()),
                            Catch::Matchers::ContainsSubstring(
                                    "mask2_front mask2_rear and barcode2_pattern must all be set"));
}

CATCH_TEST_CASE("Parse kit with no flanks", "[barcode_demux]") {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/custom_barcodes"));
    const auto test_file = data_dir / "flank_free_arrangement.toml";

    CATCH_CHECK_THROWS_WITH(
            dorado::demux::parse_custom_arrangement(test_file.string()),
            Catch::Matchers::ContainsSubstring(
                    "At least one of mask1_front or mask1_rear needs to be specified"));
}

CATCH_TEST_CASE("Parse custom barcode sequences", "[barcode_demux]") {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/custom_barcodes"));
    const auto test_sequences = data_dir / "test_sequences.fasta";

    std::unordered_map<std::string, std::string> barcode_sequences;
    auto custom_sequences = dorado::demux::parse_custom_sequences(test_sequences.string());
    for (const auto& entry : custom_sequences) {
        barcode_sequences.emplace(std::make_pair(entry.name, entry.sequence));
    }

    CATCH_CHECK(barcode_sequences.size() == 4);
    CATCH_CHECK(barcode_sequences["CUSTOM-BC01"] == "AAAAAA");
    CATCH_CHECK(barcode_sequences["CUSTOM-BC02"] == "CCCCCC");
    CATCH_CHECK(barcode_sequences["CUSTOM-BC03"] == "TTTTTT");
    CATCH_CHECK(barcode_sequences["CUSTOM-BC04"] == "GGGGGG");
}

CATCH_TEST_CASE("Parse custom barcode scoring params", "[barcode_demux]") {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/custom_barcodes"));
    const auto test_params_file = data_dir / "scoring_params.toml";

    dorado::barcode_kits::BarcodeKitScoringParams default_params;
    auto scoring_params =
            dorado::demux::parse_scoring_params(test_params_file.string(), default_params);

    CATCH_CHECK(scoring_params.max_barcode_penalty == 10);
    CATCH_CHECK(scoring_params.barcode_end_proximity == 75);
    CATCH_CHECK(scoring_params.min_barcode_penalty_dist == 3);
    CATCH_CHECK(scoring_params.min_separation_only_dist == 5);
    CATCH_CHECK(scoring_params.flank_left_pad == 5);
    CATCH_CHECK(scoring_params.flank_right_pad == 10);
    CATCH_CHECK(scoring_params.front_barcode_window == 150);
    CATCH_CHECK(scoring_params.rear_barcode_window == 150);
    CATCH_CHECK(scoring_params.min_flank_score == Catch::Approx(0.5f));
}

CATCH_TEST_CASE("Parse default scoring params", "[barcode_demux]") {
    fs::path data_dir = fs::path(get_data_dir("barcode_demux/custom_barcodes"));
    const auto test_params_file = data_dir / "test_kit_single_ended.toml";

    dorado::barcode_kits::BarcodeKitScoringParams default_params;
    auto scoring_params =
            dorado::demux::parse_scoring_params(test_params_file.string(), default_params);

    CATCH_CHECK(scoring_params.max_barcode_penalty == default_params.max_barcode_penalty);
    CATCH_CHECK(scoring_params.barcode_end_proximity == default_params.barcode_end_proximity);
    CATCH_CHECK(scoring_params.min_barcode_penalty_dist == default_params.min_barcode_penalty_dist);
    CATCH_CHECK(scoring_params.min_separation_only_dist == default_params.min_separation_only_dist);
    CATCH_CHECK(scoring_params.flank_left_pad == default_params.flank_left_pad);
    CATCH_CHECK(scoring_params.flank_right_pad == default_params.flank_right_pad);
}

CATCH_TEST_CASE("Check for normalized id pattern", "[barcode_demux]") {
    CATCH_CHECK(dorado::demux::check_normalized_id_pattern("BC%02i"));
    CATCH_CHECK(dorado::demux::check_normalized_id_pattern("abcd%25i"));
    CATCH_CHECK(dorado::demux::check_normalized_id_pattern("%2i"));

    CATCH_CHECK_FALSE(dorado::demux::check_normalized_id_pattern("ab"));
    CATCH_CHECK_FALSE(dorado::demux::check_normalized_id_pattern("ab%"));
    CATCH_CHECK_FALSE(dorado::demux::check_normalized_id_pattern("ab%d"));
    CATCH_CHECK_FALSE(dorado::demux::check_normalized_id_pattern("ab%02"));
    CATCH_CHECK_FALSE(dorado::demux::check_normalized_id_pattern("ab%02f"));
    CATCH_CHECK_FALSE(dorado::demux::check_normalized_id_pattern("ab%02iab"));
}
