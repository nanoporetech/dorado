#include "secondary/consensus/sample_collate_utils.h"

#include <catch2/catch_test_macros.hpp>
#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace dorado::secondary::sample_collate_utils::tests {

#define TEST_GROUP "[SecondaryConsensusSampleCollateUtils]"

namespace {

Sample create_mock_sample(const std::vector<int64_t>& shape) {
    if (std::empty(shape)) {
        return {};
    }
    Sample sample;
    sample.seq_id = 1;
    sample.features = torch::rand(shape);
    sample.positions_major = std::vector<int64_t>(shape.front());
    sample.positions_minor = std::vector<int64_t>(shape.front(), 0);  // All are major.
    sample.depth = torch::rand(shape.front());
    std::iota(std::begin(sample.positions_major), std::end(sample.positions_major), 0);
    return sample;
}

}  // namespace

CATCH_TEST_CASE("compute_collated_padded_shape - with new_sample, empty buffer", TEST_GROUP) {
    // Create input data.
    const std::vector<Sample> buffered_samples;
    const Sample new_sample = create_mock_sample({10000, 30, 6});

    // Expected results.
    const std::vector<int64_t> expected{1, 10000, 30, 6};

    // Run UUT.
    const std::vector<int64_t> result = compute_collated_padded_shape(buffered_samples, new_sample);

    // Eval.
    CATCH_CHECK(result == expected);
}

CATCH_TEST_CASE("compute_collated_padded_shape - with new_sample, buffer with samples",
                TEST_GROUP) {
    // Create input data.
    const std::vector<Sample> buffered_samples{
            create_mock_sample({3000, 10, 5}),
            create_mock_sample({30000, 20, 5}),
    };
    const Sample new_sample = create_mock_sample({10000, 30, 6});

    // Expected results.
    const std::vector<int64_t> expected{3, 30000, 30, 6};

    // Run UUT.
    const std::vector<int64_t> result = compute_collated_padded_shape(buffered_samples, new_sample);

    // Eval.
    CATCH_CHECK(result == expected);
}

CATCH_TEST_CASE("compute_collated_padded_shape - with new_sample, tensor shape size mismatch",
                TEST_GROUP) {
    // Create input data.
    const std::vector<Sample> buffered_samples{
            create_mock_sample({3000, 10}),  // E.g. Non read-level models
            create_mock_sample({30000, 10}),
    };
    const Sample new_sample = create_mock_sample({10000, 30, 6});  // E.g. Read-level model.

    // Eval.
    CATCH_CHECK_THROWS_AS(compute_collated_padded_shape(buffered_samples, new_sample),
                          std::runtime_error);
}

CATCH_TEST_CASE(
        "compute_collated_padded_shape - with new_sample, buffer has tensors of different shape",
        TEST_GROUP) {
    // Create input data.
    // The following buffered_samples should not even be possible, since each sample has a different shape.
    const std::vector<Sample> buffered_samples{
            create_mock_sample({3000, 10}),      // E.g. Non read-level models
            create_mock_sample({30000, 20, 5}),  // E.g. Read-level models.
    };
    const Sample new_sample = create_mock_sample({10000, 30, 6});  // E.g. Read-level model.

    // Eval.
    CATCH_CHECK_THROWS_AS(compute_collated_padded_shape(buffered_samples, new_sample),
                          std::runtime_error);
}

CATCH_TEST_CASE("compute_collated_padded_shape - with new_sample, new_sample is uninitialized",
                TEST_GROUP) {
    // Create input data.
    // Using the buffered samples shape of size 1 (one dimensional) because uninitialized torch::Tensor
    // objects have .sizes().size() == 1 for some reason (and not 0).
    // This way we find out if the function fails for the right reason.
    const std::vector<Sample> buffered_samples{
            create_mock_sample({3000}),
            create_mock_sample({30000}),
    };
    const Sample new_sample = create_mock_sample({});  // Uninitialized, should throw.

    // Eval.
    CATCH_CHECK_THROWS_AS(compute_collated_padded_shape(buffered_samples, new_sample),
                          std::runtime_error);
}

CATCH_TEST_CASE(
        "compute_collated_padded_shape - with new_sample, one buffered sample is uninitialized",
        TEST_GROUP) {
    // Create input data.
    const std::vector<Sample> buffered_samples{
            create_mock_sample({3000}), create_mock_sample({}),  // Uninitialized, should throw.
    };
    const Sample new_sample = create_mock_sample({7000});

    // Eval.
    CATCH_CHECK_THROWS_AS(compute_collated_padded_shape(buffered_samples, new_sample),
                          std::runtime_error);
}

CATCH_TEST_CASE("compute_collated_padded_shape - no new_sample, empty buffer", TEST_GROUP) {
    // Create input data.
    const std::vector<Sample> buffered_samples;

    // Expected results.
    const std::vector<int64_t> expected;

    // Run UUT.
    const std::vector<int64_t> result = compute_collated_padded_shape(buffered_samples);

    // Eval.
    CATCH_CHECK(result == expected);
}

CATCH_TEST_CASE("compute_collated_padded_shape - no new_sample, buffer with samples", TEST_GROUP) {
    // Create input data.
    const std::vector<Sample> buffered_samples{
            create_mock_sample({3000, 10, 6}),
            create_mock_sample({30000, 20, 5}),
    };

    // Expected results.
    const std::vector<int64_t> expected{2, 30000, 20, 6};

    // Run UUT.
    const std::vector<int64_t> result = compute_collated_padded_shape(buffered_samples);

    // Eval.
    CATCH_CHECK(result == expected);
}

CATCH_TEST_CASE(
        "compute_collated_padded_shape - no new_sample, buffer has tensors of different shape",
        TEST_GROUP) {
    // Create input data.
    const std::vector<Sample> buffered_samples{
            create_mock_sample({3000, 10}),  // E.g. Non read-level models
            create_mock_sample({30000, 10, 5}),
    };

    // Eval.
    CATCH_CHECK_THROWS_AS(compute_collated_padded_shape(buffered_samples), std::runtime_error);
}

CATCH_TEST_CASE(
        "compute_collated_padded_shape - no new_sample, one buffered sample is uninitialized",
        TEST_GROUP) {
    // Create input data.
    const std::vector<Sample> buffered_samples{
            create_mock_sample({3000}), create_mock_sample({}),  // Uninitialized, should throw.
    };

    // Eval.
    CATCH_CHECK_THROWS_AS(compute_collated_padded_shape(buffered_samples), std::runtime_error);
}

}  // namespace dorado::secondary::sample_collate_utils::tests