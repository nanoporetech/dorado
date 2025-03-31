#include "torch_utils/trim.h"

#include "TestUtils.h"
#include "demux/Trimmer.h"
#include "read_pipeline/HtsReader.h"
#include "read_pipeline/read_utils.h"

#include <ATen/TensorIndexing.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <htslib/sam.h>

#include <filesystem>
#include <random>
#include <string>
#include <vector>

using Catch::Matchers::Equals;
using Slice = at::indexing::Slice;
using namespace dorado;

#define TEST_GROUP "[utils][trim]"

namespace fs = std::filesystem;

CATCH_TEST_CASE("Test trim signal", TEST_GROUP) {
    constexpr int signal_len = 2000;

    std::mt19937 gen{42};
    std::normal_distribution<float> rng{0, 1};

    std::vector<float> signal(signal_len);
    std::generate(signal.begin(), signal.end(), [&]() { return rng(gen); });

    // add a peak just after the start
    for (int i = 1; i < 55; ++i) {
        signal[i] += 5;
    }

    auto signal_tensor = at::from_blob(const_cast<float *>(signal.data()), {signal_len});

    CATCH_SECTION("Default trim") {
        int pos = utils::trim(signal_tensor, utils::DEFAULT_TRIM_THRESHOLD,
                              utils::DEFAULT_TRIM_WINDOW_SIZE, utils::DEFAULT_TRIM_MIN_ELEMENTS);

        // pos 55 is in the second window of 40 samples, after a min_trim of 10
        int expected_pos = 90;
        CATCH_CHECK(pos == expected_pos);

        // begin with a plateau instead of a peak, should still find the same end
        signal[0] += 5;
        CATCH_CHECK(pos == expected_pos);
    }

    CATCH_SECTION("Reduced window size") {
        int pos = utils::trim(signal_tensor, 2.4f, 10, utils::DEFAULT_TRIM_MIN_ELEMENTS);

        int expected_pos = 60;
        CATCH_CHECK(pos == expected_pos);
    }

    CATCH_SECTION("All signal below threshold") {
        int pos = utils::trim(signal_tensor, 24, utils::DEFAULT_TRIM_WINDOW_SIZE,
                              utils::DEFAULT_TRIM_MIN_ELEMENTS);

        int expected_pos = 10;  // minimum trim value
        CATCH_CHECK(pos == expected_pos);
    }

    CATCH_SECTION("All signal above threshold") {
        std::fill(std::begin(signal), std::end(signal), 100.f);
        int pos = utils::trim(signal_tensor, 24, utils::DEFAULT_TRIM_WINDOW_SIZE,
                              utils::DEFAULT_TRIM_MIN_ELEMENTS);

        int expected_pos = 10;  // minimum trim value
        CATCH_CHECK(pos == expected_pos);
    }

    CATCH_SECTION("Peak beyond max samples") {
        for (int i = 500; i < 555; ++i) {
            signal[i] += 50;
        }

        int pos = utils::trim(signal_tensor.index({Slice(at::indexing::None, 400)}), 24,
                              utils::DEFAULT_TRIM_WINDOW_SIZE, utils::DEFAULT_TRIM_MIN_ELEMENTS);

        int expected_pos = 10;  // minimum trim value
        CATCH_CHECK(pos == expected_pos);
    }
}

CATCH_TEST_CASE("Test trim sequence", TEST_GROUP) {
    const std::string seq = "TEST_SEQ";

    CATCH_SECTION("Test empty sequence") {
        CATCH_CHECK_THROWS_AS(utils::trim_sequence("", {10, 50}), std::invalid_argument);
    }

    CATCH_SECTION("Trim nothing") {
        CATCH_CHECK(utils::trim_sequence(seq, {0, int(seq.length())}) == seq);
    }

    CATCH_SECTION("Trim part of the sequence") {
        CATCH_CHECK(utils::trim_sequence(seq, {5, int(seq.length())}) == "SEQ");
    }

    CATCH_SECTION("Trim whole sequence") { CATCH_CHECK(utils::trim_sequence(seq, {0, 0}) == ""); }
}

CATCH_TEST_CASE("Test trim quality vector", TEST_GROUP) {
    const std::vector<uint8_t> qual = {30, 30, 56, 60, 72, 10};

    CATCH_SECTION("Test empty sequence") {
        CATCH_CHECK(utils::trim_quality({}, {0, 20}).size() == 0);
    }

    CATCH_SECTION("Trim nothing") {
        CATCH_CHECK(utils::trim_quality(qual, {0, int(qual.size())}) == qual);
    }

    CATCH_SECTION("Trim part of the sequence") {
        const std::vector<uint8_t> expected = {10};
        CATCH_CHECK(utils::trim_quality(qual, {5, int(qual.size())}) == expected);
    }

    CATCH_SECTION("Trim whole sequence") {
        CATCH_CHECK(utils::trim_quality(qual, {0, 0}).size() == 0);
    }
}

CATCH_TEST_CASE("Test trim move table", TEST_GROUP) {
    const std::vector<uint8_t> move = {1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1};

    CATCH_SECTION("Trim nothing") {
        auto [ts, trimmed_table] = utils::trim_move_table(move, {0, int(move.size())});
        CATCH_CHECK(ts == 0);
        CATCH_CHECK_THAT(trimmed_table, Equals(move));
    }

    CATCH_SECTION("Trim part of the sequence") {
        auto [ts, trimmed_table] = utils::trim_move_table(move, {3, 5});
        CATCH_CHECK(ts == 6);
        const std::vector<uint8_t> expected = {1, 1, 0, 0};
        CATCH_CHECK_THAT(trimmed_table, Equals(expected));
    }

    CATCH_SECTION("Trim whole sequence") {
        auto [ts, trimmed_table] = utils::trim_move_table(move, {0, 0});
        CATCH_CHECK(ts == 0);
        CATCH_CHECK(trimmed_table.size() == 0);
    }
}

CATCH_TEST_CASE("Test trim mod base info", TEST_GROUP) {
    const std::string seq = "TAAACTTACGGTGCATCGACTG";
    const std::string modbase_str = "A+a?,2,0,1;C+m?,4;T+x?,2,2;";
    const std::vector<uint8_t> modbase_probs = {2, 3, 4, 10, 20, 21};

    CATCH_SECTION("Trim nothing") {
        auto [str, probs] =
                utils::trim_modbase_info(seq, modbase_str, modbase_probs, {0, int(seq.length())});
        CATCH_CHECK(str == modbase_str);
        CATCH_CHECK_THAT(probs, Equals(modbase_probs));
    }

    CATCH_SECTION("Trim part of the sequence") {
        // This position tests 3 cases together -
        // in the first mod, trimming truncates first 2 -> 0 and drops the last one
        // the second mod is eliminated
        // in the third mod, first base position changes and the last is dropped
        auto [str, probs] = utils::trim_modbase_info(seq, modbase_str, modbase_probs, {3, 18});
        CATCH_CHECK(str == "A+a?,0,0;C+m?;T+x?,1;");
        const std::vector<uint8_t> expected = {2, 3, 20};
        CATCH_CHECK_THAT(probs, Equals(expected));
    }

    CATCH_SECTION("Trim whole sequence") {
        auto [str, probs] = utils::trim_modbase_info(seq, modbase_str, modbase_probs, {8, 8});
        CATCH_CHECK(str == "A+a?;C+m?;T+x?;");
        CATCH_CHECK(probs.size() == 0);
    }
}

// This test case is useful because trimming of the reverse strand requires
// the sequence to be reversed, but the modbase tags are stored in the
// original sequencing direction
CATCH_TEST_CASE("Test trim of reverse strand record in BAM", TEST_GROUP) {
    const auto data_dir = fs::path(get_data_dir("trimmer"));
    const auto bam_file = data_dir / "reverse_strand_record.bam";
    HtsReader reader(bam_file.string(), std::nullopt);
    reader.read();
    auto &record = reader.record;

    Trimmer trimmer;
    const std::pair<int, int> trim_interval = {72, 647};
    auto trimmed_record = trimmer.trim_sequence(record.get(), trim_interval);
    auto seqlen = trimmed_record->core.l_qseq;

    CATCH_CHECK(seqlen == (trim_interval.second - trim_interval.first));
    CATCH_CHECK(bam_aux2i(bam_aux_get(trimmed_record.get(), "MN")) == seqlen);
    CATCH_CHECK_THAT(bam_aux2Z(bam_aux_get(trimmed_record.get(), "MM")),
                     Equals("C+h?,18,24;C+m?,18,24;"));
}

CATCH_TEST_CASE("Test trim removes all alignment information", TEST_GROUP) {
    const auto data_dir = fs::path(get_data_dir("trimmer"));
    const auto bam_file = data_dir / "reverse_strand_record.bam";
    HtsReader reader(bam_file.string(), std::nullopt);
    reader.read();
    auto &record = reader.record;

    Trimmer trimmer;
    const std::pair<int, int> trim_interval = {72, 647};
    auto trimmed_record = trimmer.trim_sequence(record.get(), trim_interval);

    CATCH_CHECK(trimmed_record->core.pos == -1);
    CATCH_CHECK(trimmed_record->core.tid == -1);
    CATCH_CHECK(trimmed_record->core.flag == 4);
    CATCH_CHECK(trimmed_record->core.n_cigar == 0);
    CATCH_CHECK(trimmed_record->core.mtid == -1);
    CATCH_CHECK(trimmed_record->core.mpos == -1);
}

static std::string to_qstr(std::vector<int8_t> qscore) {
    std::string qstr;
    for (size_t i = 0; i < qscore.size(); ++i) {
        qstr += static_cast<char>(qscore[i] + 33);
    }
    return qstr;
}

CATCH_TEST_CASE("Test find_mux_change_trim_seq_index", TEST_GROUP) {
    CATCH_SECTION("Trim simple") {
        std::vector<int8_t> vec(50, 50);
        for (size_t i = 40; i < vec.size(); ++i) {
            vec[i] = 1;
        }
        CATCH_CHECK(utils::find_mux_change_trim_seq_index(to_qstr(vec)) == 39);
    }

    CATCH_SECTION("Trim all") {
        std::vector<int8_t> vec(50, 1);
        CATCH_CHECK(utils::find_mux_change_trim_seq_index(to_qstr(vec)) == -1);
    }

    CATCH_SECTION("Trim skip single high base") {
        std::vector<int8_t> vec(50, 50);
        for (size_t i = 30; i < vec.size(); ++i) {
            vec[i] = 1;
        }
        vec[vec.size() - 1] = 50;
        CATCH_CHECK(utils::find_mux_change_trim_seq_index(to_qstr(vec)) == 29);
    }

    CATCH_SECTION("Trim nothing") {
        std::vector<int8_t> vec(120, 50);
        CATCH_CHECK(utils::find_mux_change_trim_seq_index(to_qstr(vec)) == 119);
    }
}

CATCH_TEST_CASE("Test determine_trim_interval (BC)", TEST_GROUP) {
    using std::make_tuple;
    auto [kit, top_pen, bottom_pen, top_score, bottom_score, use_top, top_pos, bottom_pos,
          expected] = GENERATE(table<std::string, int, int, float, float, bool, std::pair<int, int>,
                                     std::pair<int, int>, std::pair<int, int>>({
            {"unclassified", -1, -1, -1.f, -1.f, false, {-1, -1}, {-1, -1}, {0, 100}},
            {"good_both", 0, 0, 1.f, 1.f, false, {1, 10}, {85, 95}, {11, 85}},
            {"bad_both", 0, 0, 0.5f, 0.5f, false, {1, 10}, {85, 95}, {0, 100}},
            {"good_top", 0, 1, 1.f, 0.5f, true, {1, 10}, {85, 95}, {11, 100}},
            {"good_bottom", 1, 0, 0.5f, 1.f, false, {1, 10}, {85, 95}, {0, 85}},
            {"overlapped_top", 0, 0, 1.f, 0.7f, true, {1, 60}, {50, 95}, {61, 100}},
            {"overlapped_bottom", 0, 0, 0.7f, 1.f, false, {1, 60}, {50, 95}, {0, 50}},
            {"full_read_top", 0, 0, 0.7f, 1.f, true, {0, 100}, {60, 90}, {0, 100}},
            {"full_read_bottom", 0, 0, 0.7f, 1.f, false, {1, 60}, {0, 100}, {0, 100}},

    }));
    CATCH_CAPTURE(kit);
    dorado::BarcodeScoreResult bc_res;
    bc_res.kit = kit;
    bc_res.top_penalty = top_pen;
    bc_res.bottom_penalty = bottom_pen;
    bc_res.top_flank_score = top_score;
    bc_res.bottom_flank_score = bottom_score;
    bc_res.top_barcode_pos = top_pos;
    bc_res.bottom_barcode_pos = bottom_pos;
    bc_res.use_top = use_top;

    auto interval = dorado::Trimmer::determine_trim_interval(bc_res, 100);
    CATCH_CHECK(interval == expected);
}
