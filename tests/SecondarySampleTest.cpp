#include "secondary/consensus/sample.h"
#include "secondary/features/medaka_read_matrix_constants.h"

#include <catch2/catch_test_macros.hpp>
#include <torch/types.h>

#include <cstdint>
#include <ostream>
#include <set>
#include <utility>
#include <vector>

namespace dorado::secondary {
template <typename T>
std::ostream& operator<<(std::ostream& os, const IntervalGeneric<T>& interval) {
    os << '[' << interval.start << ", " << interval.end << ')';
    return os;
}
}  // namespace dorado::secondary

namespace dorado::secondary::sample::tests {

#define TEST_GROUP "[SecondarySample]"

namespace {

Sample make_linear_sample(const int32_t seq_id,
                          std::vector<int64_t> positions_major,
                          std::vector<int64_t> positions_minor,
                          at::Tensor depth) {
    const int64_t num_positions = std::ssize(positions_major);

    Sample sample;
    sample.seq_id = seq_id;
    sample.features = torch::arange(0, num_positions * 5, torch::kInt64).view({num_positions, 5});
    sample.positions_major = std::move(positions_major);
    sample.positions_minor = std::move(positions_minor);
    sample.depth = std::move(depth);
    sample.read_ids_left = {"left_0", "left_1"};
    sample.read_ids_right = {"right_0", "right_1"};
    return sample;
}

Sample make_linear_sample(const int32_t seq_id, std::vector<int64_t> positions_major) {
    const int64_t num_positions = std::ssize(positions_major);
    return make_linear_sample(seq_id, std::move(positions_major),
                              std::vector<int64_t>(static_cast<size_t>(num_positions), 0),
                              torch::ones({num_positions}, torch::kInt64));
}

Sample make_pileup_sample(const int32_t seq_id,
                          std::vector<int64_t> positions_major,
                          std::vector<int64_t> positions_minor,
                          const at::Tensor& depth) {
    const int64_t num_positions = std::ssize(positions_major);

    Sample sample;
    sample.seq_id = seq_id;
    sample.features = torch::zeros({num_positions, 3, 1}, torch::kInt64);
    sample.positions_major = std::move(positions_major);
    sample.positions_minor = std::move(positions_minor);
    sample.depth = depth.clone();
    sample.read_ids_left = {"left_0", "left_1", "left_2"};
    sample.read_ids_right = {"right_0", "right_1", "right_2"};
    return sample;
}

IntervalTreesInt64Map make_candidate_trees(
        const std::vector<std::pair<int32_t, std::vector<int64_t>>>& candidates_by_seq) {
    IntervalTreesInt64Map trees;
    for (const auto& [seq_id, positions] : candidates_by_seq) {
        std::vector<IntervalInt64> intervals;
        intervals.reserve(std::size(positions));
        for (const int64_t pos : positions) {
            intervals.emplace_back(pos, pos, 0);
        }
        trees.emplace(seq_id, IntervalTreeInt64(std::move(intervals)));
    }
    return trees;
}

}  // namespace

CATCH_TEST_CASE("slice_sample: Basic slicing", TEST_GROUP) {
    Sample sample;
    sample.seq_id = 1;
    sample.features = torch::tensor({{1, 2, 3, 4, 5},
                                     {6, 7, 8, 9, 10},
                                     {11, 12, 13, 14, 15},
                                     {16, 17, 18, 19, 20},
                                     {21, 22, 23, 24, 25},
                                     {26, 27, 28, 29, 30},
                                     {31, 32, 33, 34, 35},
                                     {36, 37, 38, 39, 40},
                                     {41, 42, 43, 44, 45},
                                     {46, 47, 48, 49, 50}},
                                    torch::kInt32);
    sample.positions_major = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    sample.positions_minor = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    sample.depth =
            torch::tensor({1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1}, torch::kFloat32);
    sample.draft_seq = {torch::tensor({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, torch::kInt32)};

    CATCH_SECTION("Slice middle range") {
        const int64_t idx_start = 2;
        const int64_t idx_end = 7;

        const auto expected_features = torch::tensor({{11, 12, 13, 14, 15},
                                                      {16, 17, 18, 19, 20},
                                                      {21, 22, 23, 24, 25},
                                                      {26, 27, 28, 29, 30},
                                                      {31, 32, 33, 34, 35}},
                                                     torch::kInt32);
        const auto expected_depth = torch::tensor({3.3, 4.4, 5.5, 6.6, 7.7}, torch::kFloat32);
        const std::vector<int64_t> expected_positions_major{2, 3, 4, 5, 6};
        const std::vector<int64_t> expected_positions_minor{12, 13, 14, 15, 16};
        const auto expected_draft_seq = torch::tensor({3, 4, 5, 6, 7}, torch::kInt32);

        const Sample sliced_sample = slice_sample(sample, idx_start, idx_end);

        CATCH_CHECK(sliced_sample.seq_id == sample.seq_id);
        CATCH_CHECK(sliced_sample.features.equal(expected_features));
        CATCH_CHECK(sliced_sample.depth.equal(expected_depth));
        CATCH_CHECK(sliced_sample.positions_major == expected_positions_major);
        CATCH_CHECK(sliced_sample.positions_minor == expected_positions_minor);
        CATCH_CHECK(std::empty(sliced_sample.read_ids_left));
        CATCH_CHECK(std::empty(sliced_sample.read_ids_right));
        CATCH_CHECK(sliced_sample.draft_seq->equal(expected_draft_seq));
    }

    CATCH_SECTION("Slice entire range") {
        const int64_t idx_start = 0;
        const int64_t idx_end = 10;

        const Sample sliced_sample = slice_sample(sample, idx_start, idx_end);

        CATCH_CHECK(sliced_sample.seq_id == sample.seq_id);
        CATCH_CHECK(sliced_sample.features.equal(sample.features));
        CATCH_CHECK(sliced_sample.depth.equal(sample.depth));
        CATCH_CHECK(sliced_sample.positions_major == sample.positions_major);
        CATCH_CHECK(sliced_sample.positions_minor == sample.positions_minor);
        CATCH_CHECK(std::empty(sliced_sample.read_ids_left));
        CATCH_CHECK(std::empty(sliced_sample.read_ids_right));
        CATCH_CHECK(sliced_sample.draft_seq->equal(*sample.draft_seq));
    }

    CATCH_SECTION("Slice single row") {
        const int64_t idx_start = 4;
        const int64_t idx_end = 5;

        const auto expected_features = torch::tensor({{21, 22, 23, 24, 25}}, torch::kInt32);
        const auto expected_depth = torch::tensor({5.5}, torch::kFloat32);
        const std::vector<int64_t> expected_positions_major{4};
        const std::vector<int64_t> expected_positions_minor{14};
        const auto expected_draft_seq = torch::tensor({5}, torch::kInt32);

        const Sample sliced_sample = slice_sample(sample, idx_start, idx_end);

        CATCH_CHECK(sliced_sample.seq_id == sample.seq_id);
        CATCH_CHECK(sliced_sample.features.equal(expected_features));
        CATCH_CHECK(sliced_sample.depth.equal(expected_depth));
        CATCH_CHECK(sliced_sample.positions_major == expected_positions_major);
        CATCH_CHECK(sliced_sample.positions_minor == expected_positions_minor);
        CATCH_CHECK(std::empty(sliced_sample.read_ids_left));
        CATCH_CHECK(std::empty(sliced_sample.read_ids_right));
        CATCH_CHECK(sliced_sample.draft_seq->equal(expected_draft_seq));
    }
}

CATCH_TEST_CASE("slice_sample: Preserves non-empty boundary read IDs", TEST_GROUP) {
    Sample sample = make_linear_sample(1, {0, 1, 2, 3, 4});
    const std::vector<std::string> placeholders = placeholder_read_ids(2);

    CATCH_SECTION("Interior slice uses placeholders on both sides") {
        const Sample sliced_sample = slice_sample(sample, 1, 4);

        CATCH_CHECK(sliced_sample.read_ids_left == placeholders);
        CATCH_CHECK(sliced_sample.read_ids_right == placeholders);
    }

    CATCH_SECTION("Full slice preserves both boundary ID vectors") {
        const Sample sliced_sample = slice_sample(sample, 0, 5);

        CATCH_CHECK(sliced_sample.read_ids_left == sample.read_ids_left);
        CATCH_CHECK(sliced_sample.read_ids_right == sample.read_ids_right);
    }

    CATCH_SECTION("Slice at the left boundary preserves only left IDs") {
        const Sample sliced_sample = slice_sample(sample, 0, 2);

        CATCH_CHECK(sliced_sample.read_ids_left == sample.read_ids_left);
        CATCH_CHECK(sliced_sample.read_ids_right == placeholders);
    }

    CATCH_SECTION("Slice at the right boundary preserves only right IDs") {
        const Sample sliced_sample = slice_sample(sample, 3, 5);

        CATCH_CHECK(sliced_sample.read_ids_left == placeholders);
        CATCH_CHECK(sliced_sample.read_ids_right == sample.read_ids_right);
    }
}

CATCH_TEST_CASE("slice_sample: Error conditions", TEST_GROUP) {
    Sample sample;
    sample.seq_id = 1;
    sample.features = torch::rand({10, 5});
    sample.positions_major = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    sample.positions_minor = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    sample.depth = torch::rand({10});

    CATCH_SECTION("Invalid range: idx_start >= idx_end") {
        CATCH_CHECK_THROWS_AS(slice_sample(sample, 5, 5), std::out_of_range);
        CATCH_CHECK_THROWS_AS(slice_sample(sample, 6, 5), std::out_of_range);
    }

    CATCH_SECTION("Invalid range: idx_start or idx_end out of bounds") {
        CATCH_CHECK_THROWS_AS(slice_sample(sample, -1, 5), std::out_of_range);
        CATCH_CHECK_THROWS_AS(slice_sample(sample, 0, 11), std::out_of_range);
        CATCH_CHECK_THROWS_AS(slice_sample(sample, 10, 11), std::out_of_range);
    }
}

CATCH_TEST_CASE("slice_sample: features tensor in sample not defined", TEST_GROUP) {
    Sample sample;
    sample.seq_id = 1;
    sample.positions_major = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    sample.positions_minor = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    sample.depth = torch::rand({10});

    CATCH_CHECK_THROWS_AS(slice_sample(sample, 0, 5), std::runtime_error);
}

CATCH_TEST_CASE("slice_sample: depth tensor in sample not defined", TEST_GROUP) {
    Sample sample;
    sample.seq_id = 1;
    sample.features = torch::rand({10, 5});
    sample.positions_major = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    sample.positions_minor = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

    CATCH_CHECK_THROWS_AS(slice_sample(sample, 0, 5), std::runtime_error);
}

CATCH_TEST_CASE("slice_sample: error, wrong length of the features tensor.", TEST_GROUP) {
    Sample sample;
    sample.seq_id = 1;
    sample.features = torch::rand({20, 5});
    sample.positions_major = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    sample.positions_minor = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    sample.depth = torch::rand({10});

    CATCH_CHECK_THROWS_AS(slice_sample(sample, 0, 5), std::runtime_error);
}

CATCH_TEST_CASE("slice_sample: error, wrong length of the depth tensor.", TEST_GROUP) {
    Sample sample;
    sample.seq_id = 1;
    sample.features = torch::rand({10, 5});
    sample.positions_major = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    sample.positions_minor = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    sample.depth = torch::rand({20});

    CATCH_CHECK_THROWS_AS(slice_sample(sample, 0, 5), std::runtime_error);
}

CATCH_TEST_CASE("slice_sample: error, wrong length of the positions_major vector.", TEST_GROUP) {
    Sample sample;
    sample.seq_id = 1;
    sample.features = torch::rand({10, 5});
    sample.positions_major = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    sample.positions_minor = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    sample.depth = torch::rand({10});

    CATCH_CHECK_THROWS_AS(slice_sample(sample, 0, 5), std::runtime_error);
}

CATCH_TEST_CASE("slice_sample: error, wrong length of the positions_minor vector.", TEST_GROUP) {
    Sample sample;
    sample.seq_id = 1;
    sample.features = torch::rand({10, 5});
    sample.positions_major = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    sample.positions_minor = {10, 11, 12, 13, 14, 15, 16, 17, 18};
    sample.depth = torch::rand({10});

    CATCH_CHECK_THROWS_AS(slice_sample(sample, 0, 5), std::runtime_error);
}

CATCH_TEST_CASE("Sample::validate: read IDs must match features.size(1)", TEST_GROUP) {
    const auto make_sample_with_read_ids = [](std::vector<std::string> read_ids_left,
                                              std::vector<std::string> read_ids_right) {
        Sample sample{
                .seq_id = 0,
                // read-level features: (positions, reads, channels).
                .features = at::ones({4, 3, 5}, at::kChar),
                .positions_major = {0, 1, 2, 3},
                .positions_minor = {0, 0, 0, 0},
                .depth = at::ones({4}, at::kInt),
                .read_ids_left = std::move(read_ids_left),
                .read_ids_right = std::move(read_ids_right),
        };
        return sample;
    };

    CATCH_SECTION("Both empty") {
        const Sample sample = make_sample_with_read_ids({}, {});
        CATCH_CHECK_THROWS_AS(sample.validate(), std::runtime_error);
    }

    CATCH_SECTION("Both matching") {
        const Sample sample = make_sample_with_read_ids({"l0", "l1", "l2"}, {"r0", "r1", "r2"});
        CATCH_CHECK_NOTHROW(sample.validate());
    }

    CATCH_SECTION("Only one side provided and matching") {
        const Sample sample_left = make_sample_with_read_ids({}, {"r0", "r1", "r2"});
        CATCH_CHECK_THROWS_AS(sample_left.validate(), std::runtime_error);
        const Sample sample_right = make_sample_with_read_ids({"l0", "l1", "l2"}, {});
        CATCH_CHECK_THROWS_AS(sample_right.validate(), std::runtime_error);
    }

    CATCH_SECTION("read_ids_left too short") {
        const Sample sample = make_sample_with_read_ids({"l0", "l1"}, {"r0", "r1", "r2"});
        CATCH_CHECK_THROWS_AS(sample.validate(), std::runtime_error);
    }

    CATCH_SECTION("read_ids_left too long") {
        const Sample sample =
                make_sample_with_read_ids({"l0", "l1", "l2", "l3"}, {"r0", "r1", "r2"});
        CATCH_CHECK_THROWS_AS(sample.validate(), std::runtime_error);
    }

    CATCH_SECTION("read_ids_right too short") {
        const Sample sample = make_sample_with_read_ids({"l0", "l1", "l2"}, {"r0", "r1"});
        CATCH_CHECK_THROWS_AS(sample.validate(), std::runtime_error);
    }

    CATCH_SECTION("read_ids_right too long") {
        const Sample sample =
                make_sample_with_read_ids({"l0", "l1", "l2"}, {"r0", "r1", "r2", "r3"});
        CATCH_CHECK_THROWS_AS(sample.validate(), std::runtime_error);
    }

    CATCH_SECTION("2D (counts) features do not check read IDs") {
        Sample sample = make_sample_with_read_ids({"l0"}, {});
        sample.features = torch::rand({4, 5});
        CATCH_CHECK_NOTHROW(sample.validate());
    }
}

CATCH_TEST_CASE("Sample::validate: draft_seq must match features.size(0)", TEST_GROUP) {
    const auto make_sample_with_draft = [](std::optional<at::Tensor> draft_seq) {
        Sample sample{
                .seq_id = 0,
                // read-level features: (positions, reads, channels).
                .features = at::ones({4, 2, 5}, at::kChar),
                .positions_major = {0, 1, 2, 3},
                .positions_minor = {0, 0, 0, 0},
                .depth = at::ones({4}, at::kInt),
                .read_ids_left = {"read1", "read2"},
                .read_ids_right = {"read1", "read2"},
                .draft_seq = std::move(draft_seq),
        };
        return sample;
    };

    CATCH_SECTION("draft_seq not set") {
        const Sample sample = make_sample_with_draft(std::nullopt);
        CATCH_CHECK_NOTHROW(sample.validate());
    }

    CATCH_SECTION("draft_seq empty") {
        const Sample sample = make_sample_with_draft({at::Tensor{}});
        CATCH_CHECK_NOTHROW(sample.validate());
    }

    CATCH_SECTION("draft_seq has expected length") {
        const Sample sample = make_sample_with_draft({at::ones({4}, at::kChar)});
        CATCH_CHECK_NOTHROW(sample.validate());
    }

    CATCH_SECTION("draft_seq too short") {
        const Sample sample = make_sample_with_draft({at::ones({3}, at::kChar)});
        CATCH_CHECK_THROWS_AS(sample.validate(), std::runtime_error);
    }

    CATCH_SECTION("draft_seq too long") {
        const Sample sample = make_sample_with_draft({at::ones({5}, at::kChar)});
        CATCH_CHECK_THROWS_AS(sample.validate(), std::runtime_error);
    }

    CATCH_SECTION("draft_seq wrong dimension") {
        const Sample sample = make_sample_with_draft({at::ones({4, 1}, at::kChar)});
        CATCH_CHECK_THROWS_AS(sample.validate(), std::runtime_error);
    }
}

CATCH_TEST_CASE("find_max_depth", TEST_GROUP) {
    Sample sample;
    sample.seq_id = 1;
    sample.features = torch::tensor({{1, 2, 3, 4, 5},
                                     {6, 7, 8, 9, 10},
                                     {11, 12, 13, 14, 15},
                                     {16, 17, 18, 19, 20},
                                     {21, 22, 23, 24, 25},
                                     {26, 27, 28, 29, 30},
                                     {31, 32, 33, 34, 35},
                                     {36, 37, 38, 39, 40},
                                     {41, 42, 43, 44, 45},
                                     {46, 47, 48, 49, 50}},
                                    torch::kInt32);
    sample.positions_major = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    sample.positions_minor = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    sample.depth = torch::tensor({1, 2, 2, 3, 2, 5, 2, 1, 0, 11}, torch::kInt64);

    CATCH_SECTION("Throws. Entire range") {
        const int64_t idx_start = 0;
        const int64_t idx_end = 10;

        constexpr int64_t EXPECTED = 11;

        const int64_t result = sample.find_max_depth(idx_start, idx_end);

        CATCH_CHECK(result == EXPECTED);
    }

    CATCH_SECTION("Subrange") {
        const int64_t idx_start = 6;
        const int64_t idx_end = 9;

        constexpr int64_t EXPECTED = 2;

        const int64_t result = sample.find_max_depth(idx_start, idx_end);

        CATCH_CHECK(result == EXPECTED);
    }

    CATCH_SECTION(
            "Throws. Float depth data type. This should throw because depth should be integral.") {
        Sample sample2 = sample;
        sample2.depth = torch::tensor({1.0f, 2.0f, 2.0f, 3.0f, 2.0f, 5.0f, 2.0f, 1.0f, 0.0f, 11.0f},
                                      torch::kFloat);
        const int64_t idx_start = 0;
        const int64_t idx_end = 10;

        CATCH_CHECK_THROWS(sample2.find_max_depth(idx_start, idx_end));
    }

    CATCH_SECTION("Throws. Start index < 0") {
        const int64_t idx_start = -1;
        const int64_t idx_end = 9;

        CATCH_CHECK_THROWS(sample.find_max_depth(idx_start, idx_end));
    }

    CATCH_SECTION("Throws. Emd index < start index") {
        const int64_t idx_start = 5;
        const int64_t idx_end = 4;

        CATCH_CHECK_THROWS(sample.find_max_depth(idx_start, idx_end));
    }

    CATCH_SECTION("Throws. Emd index == start index") {
        const int64_t idx_start = 5;
        const int64_t idx_end = 5;

        CATCH_CHECK_THROWS(sample.find_max_depth(idx_start, idx_end));
    }

    CATCH_SECTION("Throws. End index > length.") {
        const int64_t idx_start = 5;
        const int64_t idx_end = 11;

        CATCH_CHECK_THROWS(sample.find_max_depth(idx_start, idx_end));
    }
}

CATCH_TEST_CASE("split_samples: overlapping windows and edge cases", TEST_GROUP) {
    CATCH_SECTION("Empty input returns no samples") {
        const std::vector<Sample> result = split_samples({}, 4, 1);

        CATCH_CHECK(std::empty(result));
    }

    CATCH_SECTION("Long samples are split with overlap and a full-length trailing chunk") {
        const Sample sample = make_linear_sample(21, {0, 1, 2, 3, 4, 5, 6, 7, 8});

        const std::vector<Sample> result = split_samples({sample}, 4, 1);

        CATCH_REQUIRE(std::size(result) == 3);
        CATCH_CHECK(result[0].positions_major == std::vector<int64_t>{0, 1, 2, 3});
        CATCH_CHECK(result[1].positions_major == std::vector<int64_t>{3, 4, 5, 6});
        CATCH_CHECK(result[2].positions_major == std::vector<int64_t>{5, 6, 7, 8});
        CATCH_CHECK(result[0].features.equal(torch::tensor(
                {{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}, {10, 11, 12, 13, 14}, {15, 16, 17, 18, 19}},
                torch::kInt64)));
        CATCH_CHECK(result[1].features.equal(torch::tensor({{15, 16, 17, 18, 19},
                                                            {20, 21, 22, 23, 24},
                                                            {25, 26, 27, 28, 29},
                                                            {30, 31, 32, 33, 34}},
                                                           torch::kInt64)));
        CATCH_CHECK(result[1].depth.equal(torch::ones({4}, torch::kInt64)));
        CATCH_CHECK(result[2].features.equal(torch::tensor({{25, 26, 27, 28, 29},
                                                            {30, 31, 32, 33, 34},
                                                            {35, 36, 37, 38, 39},
                                                            {40, 41, 42, 43, 44}},
                                                           torch::kInt64)));
    }

    CATCH_SECTION("Samples at or below chunk length are preserved") {
        const Sample sample = make_linear_sample(22, {10, 11, 12, 13});

        const std::vector<Sample> result = split_samples({sample}, 4, 1);

        CATCH_REQUIRE(std::size(result) == 1);
        CATCH_CHECK(result[0].positions_major == sample.positions_major);
        CATCH_CHECK(result[0].positions_minor == sample.positions_minor);
        CATCH_CHECK(result[0].features.equal(sample.features));
        CATCH_CHECK(result[0].depth.equal(sample.depth));
    }

    CATCH_SECTION("Invalid overlap throws") {
        const Sample sample = make_linear_sample(23, {0, 1, 2, 3, 4});

        CATCH_CHECK_THROWS_AS(split_samples({sample}, 4, -1), std::runtime_error);
        CATCH_CHECK_THROWS_AS(split_samples({sample}, 4, 5), std::runtime_error);
    }

    CATCH_SECTION("Invalid chunk length throws") {
        const Sample sample = make_linear_sample(24, {0, 1, 2, 3, 4});

        CATCH_CHECK_THROWS_AS(split_samples({sample}, 0, 0), std::runtime_error);
        CATCH_CHECK_THROWS_AS(split_samples({sample}, -1, 0), std::runtime_error);
    }
}

CATCH_TEST_CASE("split_samples_around_positions: candidate-driven chunking and edge cases",
                TEST_GROUP) {
    const Sample sample = make_linear_sample(31, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

    CATCH_SECTION("Empty sample input returns no chunks") {
        const IntervalTreesInt64Map candidate_trees = make_candidate_trees({{31, {4}}});

        const std::vector<Sample> result =
                split_samples_around_positions({}, candidate_trees, 7, 3);

        CATCH_CHECK(std::empty(result));
    }

    CATCH_SECTION("Empty candidate trees return no chunks") {
        const std::vector<Sample> result = split_samples_around_positions({sample}, {}, 7, 3);

        CATCH_CHECK(std::empty(result));
    }

    CATCH_SECTION(
            "Chunks are created around candidate positions, duplicates are skipped, and "
            "late overhangs stop chunking") {
        const Sample skipped_sample = make_linear_sample(32, {20, 21, 22, 23, 24, 25, 26, 27});
        const IntervalTreesInt64Map candidate_trees =
                make_candidate_trees({{31, {4, 4, 8}}, {999, {100}}});

        const std::vector<Sample> result =
                split_samples_around_positions({sample, skipped_sample}, candidate_trees, 7, 3);

        CATCH_REQUIRE(std::size(result) == 1);
        CATCH_CHECK(result[0].seq_id == 31);
        CATCH_CHECK(result[0].positions_major == std::vector<int64_t>{1, 2, 3, 4, 5, 6, 7});
        CATCH_CHECK(result[0].positions_minor == std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0});
        CATCH_CHECK(result[0].features.equal(torch::tensor({{5, 6, 7, 8, 9},
                                                            {10, 11, 12, 13, 14},
                                                            {15, 16, 17, 18, 19},
                                                            {20, 21, 22, 23, 24},
                                                            {25, 26, 27, 28, 29},
                                                            {30, 31, 32, 33, 34},
                                                            {35, 36, 37, 38, 39}},
                                                           torch::kInt64)));
        CATCH_CHECK(result[0].depth.equal(torch::ones({7}, torch::kInt64)));
        CATCH_CHECK(result[0].read_ids_left == placeholder_read_ids(2));
        CATCH_CHECK(result[0].read_ids_right == placeholder_read_ids(2));
    }

    CATCH_SECTION("Invalid flanking_bases throws") {
        const IntervalTreesInt64Map candidate_trees = make_candidate_trees({{31, {4}}});

        CATCH_CHECK_THROWS_AS(split_samples_around_positions({sample}, candidate_trees, 7, -1),
                              std::runtime_error);
        CATCH_CHECK_THROWS_AS(split_samples_around_positions({sample}, candidate_trees, 7, 8),
                              std::runtime_error);
    }

    CATCH_SECTION("Invalid chunk length throws") {
        const IntervalTreesInt64Map candidate_trees = make_candidate_trees({{31, {4}}});

        CATCH_CHECK_THROWS_AS(split_samples_around_positions({sample}, candidate_trees, 0, 0),
                              std::runtime_error);
        CATCH_CHECK_THROWS_AS(split_samples_around_positions({sample}, candidate_trees, -1, 0),
                              std::runtime_error);
    }
}

CATCH_TEST_CASE("split_samples_tiled_with_candidates: flank extension is applied only when needed",
                TEST_GROUP) {
    /**
     * \brief 60 linear columns (major = column index, no insertions), chunk_len = 6 and chunk_overlap = 2,
     *          so the windows start every 4 columns: [0,6) [4,10) [8,14) [12,18) [16,22) [20,26) ...
     *          The only candidate is at column 15, which is contained only in the window [12,18).
     *          The deletion check looks at the first/last 2 major positions of a window and requires
     *          at least max(2, 0.5 * max depth) reads with a deletion there (all 3 reads here).
     */
    constexpr int64_t NUM_COLUMNS = 60;

    std::vector<int64_t> positions_major(NUM_COLUMNS);
    for (int64_t i = 0; i < NUM_COLUMNS; ++i) {
        positions_major[i] = i;
    }

    const IntervalTreesInt64Map candidate_trees = make_candidate_trees({{47, {15}}});

    const auto make_sample = [&positions_major]() {
        return make_pileup_sample(47, positions_major, std::vector<int64_t>(NUM_COLUMNS, 0),
                                  torch::full({NUM_COLUMNS}, 3, torch::kInt64));
    };

    // Returns the first major position of each returned window.
    const auto window_starts = [](const std::vector<Sample>& windows) {
        std::vector<int64_t> starts;
        starts.reserve(std::size(windows));
        for (const Sample& window : windows) {
            starts.emplace_back(window.positions_major.front());
        }
        return starts;
    };

    CATCH_SECTION("No insertions and no deletions at the flanks: nothing is extended") {
        const Sample sample = make_sample();

        const std::vector<Sample> without_extension = split_samples_tiled_with_candidates(
                {sample}, candidate_trees, 6, 2, false, 2, 2, 0.5);
        const std::vector<Sample> with_extension = split_samples_tiled_with_candidates(
                {sample}, candidate_trees, 6, 2, true, 2, 2, 0.5);

        CATCH_CHECK(window_starts(without_extension) == std::vector<int64_t>{12});
        CATCH_CHECK(window_starts(with_extension) == std::vector<int64_t>{12});
    }

    CATCH_SECTION("Deletions at the end flank extend only to the right") {
        // Columns 16 and 17 are the last 2 major positions of the window [12,18).
        Sample sample = make_sample();
        sample.features.index_put_({torch::indexing::Slice(16, 18), torch::indexing::Slice(), 0},
                                   DEL_VAL);

        const std::vector<Sample> result = split_samples_tiled_with_candidates(
                {sample}, candidate_trees, 6, 2, true, 2, 2, 0.5);

        // The extension window [16,22) has a clean end flank, so it does not extend any further.
        CATCH_CHECK(window_starts(result) == std::vector<int64_t>{12, 16});
    }

    CATCH_SECTION("Deletions at the start flank extend only to the left") {
        // Columns 12 and 13 are the first 2 major positions of the window [12,18).
        Sample sample = make_sample();
        sample.features.index_put_({torch::indexing::Slice(12, 14), torch::indexing::Slice(), 0},
                                   DEL_VAL);

        const std::vector<Sample> result = split_samples_tiled_with_candidates(
                {sample}, candidate_trees, 6, 2, true, 2, 2, 0.5);

        // The extension window [8,14) has a clean start flank, so it does not extend any further.
        CATCH_CHECK(window_starts(result) == std::vector<int64_t>{8, 12});
    }

    CATCH_SECTION("Deletions at the end flank of an extension window continue the extension") {
        // Deletions at the end flanks of both [12,18) (columns 16,17) and [16,22) (columns 20,21).
        Sample sample = make_sample();
        sample.features.index_put_({torch::indexing::Slice(16, 18), torch::indexing::Slice(), 0},
                                   DEL_VAL);
        sample.features.index_put_({torch::indexing::Slice(20, 22), torch::indexing::Slice(), 0},
                                   DEL_VAL);

        const std::vector<Sample> result = split_samples_tiled_with_candidates(
                {sample}, candidate_trees, 6, 2, true, 2, 2, 0.5);

        CATCH_CHECK(window_starts(result) == std::vector<int64_t>{12, 16, 20});
    }
}

CATCH_TEST_CASE("split_samples_tiled_with_candidates: windows cover an entire long insertion",
                TEST_GROUP) {
    /**
     * \brief Mimics a real case (chr20:26589669) where an insertion longer than several windows was
     *          only partially selected for inference. The insertion after major 5 is not a candidate
     *          itself; the candidates are only on its flanks (majors 4 and 9).
     *
     *          Columns (chunk_len = 6, chunk_overlap = 2, so windows start every 4 columns):
     *            - IDX  :  0 1 2 3 4 5 | 6 .. 17      | 18 19 20 21 22 23
     *            - MAJOR:  0 1 2 3 4 5 | 5 .. 5       |  6  7  8  9 10 11
     *            - MINOR:  0 0 0 0 0 0 | 1 .. 12      |  0  0  0  0  0  0
     *          Windows: [0,6) [4,10) [8,14) [12,18) [16,22) [18,24).
     *          The windows [0,6) and [4,10) contain the candidate at major 4, and [16,22) and [18,24)
     *          contain the candidate at major 9. The windows [8,14) and [12,18) contain only the
     *          insertion (major 5) and have no candidate, so they are not selected and the middle of
     *          the insertion (minors 5 to 10) is missing from inference.
     *          Selected windows which end or begin inside an insertion should be extended to the
     *          neighboring windows until the entire insertion is covered.
     */
    std::vector<int64_t> positions_major{0, 1, 2, 3, 4, 5};
    std::vector<int64_t> positions_minor{0, 0, 0, 0, 0, 0};
    for (int64_t minor = 1; minor <= 12; ++minor) {
        positions_major.emplace_back(5);
        positions_minor.emplace_back(minor);
    }
    for (int64_t major = 6; major <= 11; ++major) {
        positions_major.emplace_back(major);
        positions_minor.emplace_back(0);
    }
    const int64_t num_columns = std::ssize(positions_major);

    // No deletions in the pileup, so the deletion-based flank extension does not apply.
    const Sample sample = make_pileup_sample(46, positions_major, positions_minor,
                                             torch::full({num_columns}, 3, torch::kInt64));
    const IntervalTreesInt64Map candidate_trees = make_candidate_trees({{46, {4, 9}}});

    const std::vector<Sample> result =
            split_samples_tiled_with_candidates({sample}, candidate_trees, 6, 2, true, 2, 2, 0.5);

    CATCH_CHECK(std::size(result) == 6);

    // Collect all columns covered by the selected windows.
    std::set<std::pair<int64_t, int64_t>> covered;
    for (const Sample& window : result) {
        for (int64_t i = 0; i < std::ssize(window.positions_major); ++i) {
            covered.emplace(window.positions_major[i], window.positions_minor[i]);
        }
    }

    // Every insertion column is covered by at least one window.
    for (int64_t minor = 1; minor <= 12; ++minor) {
        CATCH_CAPTURE(minor);
        CATCH_CHECK(covered.count({5, minor}) == 1);
    }
}

CATCH_TEST_CASE("split_samples_tiled_with_candidates: candidate filtering and flank extension",
                TEST_GROUP) {
    CATCH_SECTION("Empty sample input returns no chunks") {
        const IntervalTreesInt64Map candidate_trees = make_candidate_trees({{41, {11}}});

        const std::vector<Sample> result =
                split_samples_tiled_with_candidates({}, candidate_trees, 4, 1, false, 1, 1, 0.5);

        CATCH_CHECK(std::empty(result));
    }

    CATCH_SECTION("Empty candidate trees return no chunks") {
        const Sample sample = make_linear_sample(40, {10, 11, 12, 13});

        const std::vector<Sample> result =
                split_samples_tiled_with_candidates({sample}, {}, 4, 1, false, 1, 1, 0.5);

        CATCH_CHECK(std::empty(result));
    }

    CATCH_SECTION("Short samples are kept only when they overlap a candidate") {
        const Sample sample = make_linear_sample(41, {10, 11, 12});
        const IntervalTreesInt64Map hit_trees = make_candidate_trees({{41, {11}}});
        const IntervalTreesInt64Map miss_trees = make_candidate_trees({{41, {20}}});

        const std::vector<Sample> hit =
                split_samples_tiled_with_candidates({sample}, hit_trees, 4, 1, false, 1, 1, 0.5);
        const std::vector<Sample> miss =
                split_samples_tiled_with_candidates({sample}, miss_trees, 4, 1, false, 1, 1, 0.5);

        CATCH_REQUIRE(std::size(hit) == 1);
        CATCH_CHECK(hit[0].positions_major == sample.positions_major);
        CATCH_CHECK(hit[0].features.equal(sample.features));
        CATCH_CHECK(std::empty(miss));
    }

    CATCH_SECTION("Only tiled windows with candidates are returned") {
        const Sample sample = make_linear_sample(42, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        const IntervalTreesInt64Map candidate_trees = make_candidate_trees({{42, {1, 8}}});

        const std::vector<Sample> result = split_samples_tiled_with_candidates(
                {sample}, candidate_trees, 4, 1, false, 1, 1, 0.5);

        CATCH_REQUIRE(std::size(result) == 2);
        CATCH_CHECK(result[0].positions_major == std::vector<int64_t>{0, 1, 2, 3});
        CATCH_CHECK(result[1].positions_major == std::vector<int64_t>{6, 7, 8, 9});
    }

    CATCH_SECTION("Deletion-heavy minor flanks extend selection to neighboring windows") {
        Sample sample = make_pileup_sample(43, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                           {0, 0, 0, 1, 0, 0, 1, 0, 0, 0},
                                           torch::full({10}, 3, torch::kInt64));
        sample.features.index_put_({4, torch::indexing::Slice(), 0}, 5);
        sample.features.index_put_({5, torch::indexing::Slice(), 0}, 5);

        const IntervalTreesInt64Map candidate_trees = make_candidate_trees({{43, {5}}});

        const std::vector<Sample> baseline = split_samples_tiled_with_candidates(
                {sample}, candidate_trees, 4, 1, false, 2, 2, 0.5);
        const std::vector<Sample> extended = split_samples_tiled_with_candidates(
                {sample}, candidate_trees, 4, 1, true, 2, 2, 0.5);

        CATCH_REQUIRE(std::size(baseline) == 1);
        CATCH_CHECK(baseline[0].positions_major == std::vector<int64_t>{3, 4, 5, 6});

        CATCH_REQUIRE(std::size(extended) == 3);
        CATCH_CHECK(extended[0].positions_major == std::vector<int64_t>{0, 1, 2, 3});
        CATCH_CHECK(extended[1].positions_major == std::vector<int64_t>{3, 4, 5, 6});
        CATCH_CHECK(extended[2].positions_major == std::vector<int64_t>{6, 7, 8, 9});
    }

    CATCH_SECTION("Invalid overlap throws") {
        const Sample sample = make_linear_sample(44, {0, 1, 2, 3, 4, 5});
        const IntervalTreesInt64Map candidate_trees = make_candidate_trees({{44, {2}}});

        CATCH_CHECK_THROWS_AS(split_samples_tiled_with_candidates({sample}, candidate_trees, 4, -1,
                                                                  false, 1, 1, 0.5),
                              std::runtime_error);
        CATCH_CHECK_THROWS_AS(split_samples_tiled_with_candidates({sample}, candidate_trees, 4, 5,
                                                                  false, 1, 1, 0.5),
                              std::runtime_error);
    }

    CATCH_SECTION("Invalid chunk length throws") {
        const Sample sample = make_linear_sample(45, {0, 1, 2, 3, 4, 5});
        const IntervalTreesInt64Map candidate_trees = make_candidate_trees({{45, {2}}});

        CATCH_CHECK_THROWS_AS(split_samples_tiled_with_candidates({sample}, candidate_trees, 0, 0,
                                                                  false, 1, 1, 0.5),
                              std::runtime_error);
        CATCH_CHECK_THROWS_AS(split_samples_tiled_with_candidates({sample}, candidate_trees, -1, 0,
                                                                  false, 1, 1, 0.5),
                              std::runtime_error);
    }
}

CATCH_TEST_CASE("find_sample_intervals: splits on true gaps and excludes low-depth runs",
                TEST_GROUP) {
    Sample sample;
    sample.seq_id = 1;
    sample.features = torch::rand({8, 5});
    sample.positions_major = {0, 1, 1, 2, 5, 6, 6, 7};
    sample.positions_minor = {0, 0, 1, 0, 0, 0, 1, 0};
    sample.read_ids_left = {"left_0", "left_1"};
    sample.read_ids_right = {"right_0", "right_1"};
    sample.depth = torch::tensor({5, 4, 1, 0, 4, 5, 2, 6}, torch::kInt64);

    CATCH_SECTION("Low-depth runs are excluded from the kept intervals") {
        const std::vector<Interval64> result = find_sample_intervals(sample, true, 3);
        const std::vector<Interval64> expected = {
                {0, 2},
                {4, 6},
                {7, 8},
        };

        CATCH_CHECK(result == expected);
    }

    CATCH_SECTION("True coordinate gaps split intervals even without low coverage") {
        const std::vector<Interval64> result = find_sample_intervals(sample, true, -1);
        const std::vector<Interval64> expected = {
                {0, 4},
                {4, 8},
        };

        CATCH_CHECK(result == expected);
    }
}

CATCH_TEST_CASE("find_sample_intervals: edge cases", TEST_GROUP) {
    CATCH_SECTION("Empty but valid sample returns no intervals") {
        Sample sample;
        sample.seq_id = 11;
        sample.features = torch::empty({0, 5}, torch::kFloat32);
        sample.positions_major = {};
        sample.positions_minor = {};
        sample.depth = torch::empty({0}, torch::kInt64);

        const std::vector<Interval64> result = find_sample_intervals(sample, true, 3);

        CATCH_CHECK(std::empty(result));
    }

    CATCH_SECTION("Dense input without gaps returns one full interval") {
        Sample sample;
        sample.seq_id = 12;
        sample.features = torch::rand({6, 5});
        sample.positions_major = {10, 10, 11, 11, 12, 13};
        sample.positions_minor = {0, 1, 0, 1, 0, 0};
        sample.depth = torch::tensor({3, 4, 5, 6, 7, 8}, torch::kInt64);

        const std::vector<Interval64> result = find_sample_intervals(sample, true, 3);

        CATCH_REQUIRE(std::size(result) == 1);
        CATCH_CHECK(result[0].start == 0);
        CATCH_CHECK(result[0].end == 6);
    }

    CATCH_SECTION("Everything masked out returns no intervals") {
        Sample sample;
        sample.seq_id = 13;
        sample.features = torch::rand({4, 5});
        sample.positions_major = {0, 1, 2, 3};
        sample.positions_minor = {0, 0, 0, 0};
        sample.depth = torch::tensor({0, 1, 1, 2}, torch::kInt64);

        const std::vector<Interval64> result = find_sample_intervals(sample, true, 3);

        CATCH_CHECK(std::empty(result));
    }

    CATCH_SECTION("Coordinate gaps are ignored when splitting is disabled") {
        Sample sample;
        sample.seq_id = 14;
        sample.features = torch::rand({4, 5});
        sample.positions_major = {0, 1, 5, 6};
        sample.positions_minor = {0, 0, 0, 0};
        sample.depth = torch::tensor({5, 5, 5, 5}, torch::kInt64);

        const std::vector<Interval64> result = find_sample_intervals(sample, false, 1);

        CATCH_REQUIRE(std::size(result) == 1);
        CATCH_CHECK(result[0].start == 0);
        CATCH_CHECK(result[0].end == 4);
    }
}

CATCH_TEST_CASE("split_sample_on_intervals: preserves flanking read IDs and slices correctly",
                TEST_GROUP) {
    Sample sample;
    sample.seq_id = 7;
    sample.features = torch::arange(0, 40, torch::kInt64).view({8, 5});
    sample.positions_major = {0, 1, 1, 2, 5, 6, 6, 7};
    sample.positions_minor = {0, 0, 1, 0, 0, 0, 1, 0};
    sample.depth = torch::tensor({5, 4, 1, 0, 4, 5, 2, 6}, torch::kInt64);
    sample.read_ids_left = {"left_0", "left_1"};
    sample.read_ids_right = {"right_0", "right_1"};

    const std::vector<Interval64> intervals = {
            {0, 2},
            {4, 6},
            {7, 8},
    };

    const std::vector<Sample> result = split_sample_on_intervals(sample, intervals);

    CATCH_REQUIRE(std::size(result) == 3);

    CATCH_CHECK(result[0].features.equal(
            torch::tensor({{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}}, torch::kInt64)));
    CATCH_CHECK(result[1].features.equal(
            torch::tensor({{20, 21, 22, 23, 24}, {25, 26, 27, 28, 29}}, torch::kInt64)));
    CATCH_CHECK(result[2].features.equal(torch::tensor({{35, 36, 37, 38, 39}}, torch::kInt64)));

    CATCH_CHECK(result[0].positions_major == std::vector<int64_t>{0, 1});
    CATCH_CHECK(result[1].positions_major == std::vector<int64_t>{5, 6});
    CATCH_CHECK(result[2].positions_major == std::vector<int64_t>{7});

    CATCH_CHECK(result[0].read_ids_left == sample.read_ids_left);
    CATCH_CHECK(result[0].read_ids_right ==
                std::vector<std::string>{"__placeholder_0", "__placeholder_1"});
    CATCH_CHECK(result[1].read_ids_left ==
                std::vector<std::string>{"__placeholder_0", "__placeholder_1"});
    CATCH_CHECK(result[1].read_ids_right ==
                std::vector<std::string>{"__placeholder_0", "__placeholder_1"});
    CATCH_CHECK(result[2].read_ids_left ==
                std::vector<std::string>{"__placeholder_0", "__placeholder_1"});
    CATCH_CHECK(result[2].read_ids_right == sample.read_ids_right);
}

CATCH_TEST_CASE("split_sample_on_intervals: edge cases", TEST_GROUP) {
    Sample sample;
    sample.seq_id = 15;
    sample.features = torch::arange(0, 20, torch::kInt64).view({4, 5});
    sample.positions_major = {0, 1, 2, 3};
    sample.positions_minor = {0, 0, 0, 0};
    sample.depth = torch::tensor({5, 5, 5, 5}, torch::kInt64);
    sample.read_ids_left = {"left_0", "left_1"};
    sample.read_ids_right = {"right_0", "right_1"};

    CATCH_SECTION("Empty interval list returns no samples") {
        const std::vector<Sample> result = split_sample_on_intervals(sample, {});

        CATCH_CHECK(std::empty(result));
    }

    CATCH_SECTION("Full interval returns one sample with original boundary read IDs") {
        const std::vector<Sample> result = split_sample_on_intervals(sample, {{0, 4}});

        CATCH_REQUIRE(std::size(result) == 1);
        CATCH_CHECK(result[0].features.equal(sample.features));
        CATCH_CHECK(result[0].positions_major == sample.positions_major);
        CATCH_CHECK(result[0].positions_minor == sample.positions_minor);
        CATCH_CHECK(result[0].depth.equal(sample.depth));
        CATCH_CHECK(result[0].read_ids_left == sample.read_ids_left);
        CATCH_CHECK(result[0].read_ids_right == sample.read_ids_right);
    }

    CATCH_SECTION("Negative interval start throws") {
        CATCH_CHECK_THROWS_AS(split_sample_on_intervals(sample, {{-1, 2}}), std::runtime_error);
    }

    CATCH_SECTION("Empty interval throws") {
        CATCH_CHECK_THROWS_AS(split_sample_on_intervals(sample, {{2, 2}}), std::runtime_error);
    }

    CATCH_SECTION("Out of bounds interval throws") {
        CATCH_CHECK_THROWS_AS(split_sample_on_intervals(sample, {{0, 5}}), std::runtime_error);
    }
}

CATCH_TEST_CASE("split_sample_on_discontinuities: splits features and preserves boundary IDs",
                TEST_GROUP) {
    Sample sample;
    sample.seq_id = 16;
    sample.features = torch::arange(0, 40, torch::kInt64).view({8, 5});
    sample.positions_major = {0, 1, 1, 2, 5, 6, 6, 7};
    sample.positions_minor = {0, 0, 1, 0, 0, 0, 1, 0};
    sample.depth = torch::tensor({4, 4, 4, 4, 4, 4, 4, 4}, torch::kInt64);
    sample.read_ids_left = {"left_0", "left_1"};
    sample.read_ids_right = {"right_0", "right_1"};

    const std::vector<Sample> result = split_sample_on_discontinuities(sample);

    CATCH_REQUIRE(std::size(result) == 2);

    CATCH_CHECK(result[0].features.equal(torch::tensor(
            {{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}, {10, 11, 12, 13, 14}, {15, 16, 17, 18, 19}},
            torch::kInt64)));
    CATCH_CHECK(result[1].features.equal(torch::tensor({{20, 21, 22, 23, 24},
                                                        {25, 26, 27, 28, 29},
                                                        {30, 31, 32, 33, 34},
                                                        {35, 36, 37, 38, 39}},
                                                       torch::kInt64)));

    CATCH_CHECK(result[0].positions_major == std::vector<int64_t>{0, 1, 1, 2});
    CATCH_CHECK(result[1].positions_major == std::vector<int64_t>{5, 6, 6, 7});

    CATCH_CHECK(result[0].read_ids_left == sample.read_ids_left);
    CATCH_CHECK(result[0].read_ids_right ==
                std::vector<std::string>{"__placeholder_0", "__placeholder_1"});
    CATCH_CHECK(result[1].read_ids_left ==
                std::vector<std::string>{"__placeholder_0", "__placeholder_1"});
    CATCH_CHECK(result[1].read_ids_right == sample.read_ids_right);
}

CATCH_TEST_CASE("split_sample_on_discontinuities: edge cases", TEST_GROUP) {
    CATCH_SECTION("Dense input is returned unsplit") {
        Sample sample;
        sample.seq_id = 16;
        sample.features = torch::arange(0, 30, torch::kInt64).view({6, 5});
        sample.positions_major = {20, 20, 21, 21, 22, 23};
        sample.positions_minor = {0, 1, 0, 1, 0, 0};
        sample.depth = torch::tensor({4, 4, 4, 4, 4, 4}, torch::kInt64);
        sample.read_ids_left = {"left_0", "left_1"};
        sample.read_ids_right = {"right_0", "right_1"};

        const std::vector<Sample> result = split_sample_on_discontinuities(sample);

        CATCH_REQUIRE(std::size(result) == 1);
        CATCH_CHECK(result[0].features.equal(sample.features));
        CATCH_CHECK(result[0].positions_major == sample.positions_major);
        CATCH_CHECK(result[0].positions_minor == sample.positions_minor);
        CATCH_CHECK(result[0].depth.equal(sample.depth));
        CATCH_CHECK(result[0].read_ids_left == sample.read_ids_left);
        CATCH_CHECK(result[0].read_ids_right == sample.read_ids_right);
    }

    CATCH_SECTION("Empty but valid sample returns no samples") {
        Sample sample;
        sample.seq_id = 17;
        sample.features = torch::empty({0, 5}, torch::kFloat32);
        sample.positions_major = {};
        sample.positions_minor = {};
        sample.depth = torch::empty({0}, torch::kInt64);

        const std::vector<Sample> result = split_sample_on_discontinuities(sample);

        CATCH_CHECK(std::empty(result));
    }
}

}  // namespace dorado::secondary::sample::tests
