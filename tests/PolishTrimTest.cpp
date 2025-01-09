#include "polish/trim.h"

#include <catch2/catch.hpp>

#include <cstdint>
#include <optional>
#include <vector>

namespace {

#define TEST_GROUP "[PolishTrimSamples]"

TEST_CASE("trim_samples tests", TEST_GROUP) {
    using namespace dorado::polisher;

    // The actual Sample struct is complicated, but the only thing needed
    // for trimming are seq_id, positions_major and positions_minor.
    // This would not be needed if C++20 was available becauase we could simply use
    // designited initializers.
    struct MockSample {
        int32_t seq_id = -1;
        std::vector<int64_t> positions_major;
        std::vector<int64_t> positions_minor;
    };

    struct TestCase {
        std::string name;                       // Name of the test.
        std::vector<MockSample> samples;        // Input samples to trim.
        std::optional<const RegionInt> region;  // Region from which the samples were generated.
        bool expect_throw = false;
        std::vector<TrimInfo> expected;  // Expected results.
    };

    // clang-format off
    auto [test_case] = GENERATE(table<TestCase>({
        TestCase{
            "Empty input == empty output",
            {},             // samples
            std::nullopt,   // region
            false,          // expect_throw
            {},             // expected
        },
        TestCase{
            "Single sample",
            {   // Input samples.
                MockSample{
                    0,                              // seq_id
                    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, // positions_major
                    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // positions_minor
                },
            },
            RegionInt{0, 0, 10},   // region
            false,              // expect_throw
            {   // Expected results.
                TrimInfo{0, 10, false},
            },
        },
        TestCase{
            "Medaka test. Three samples, overlapping. Should trim.",
            {   // Input samples.
                MockSample{
                    0,                                  // seq_id
                    {0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 4},  // positions_major
                    {0, 1, 0, 0, 1, 2, 0, 0, 1, 2, 3},  // positions_minor
                },
                MockSample{
                    0,                         // seq_id
                    {4, 4, 4, 4, 5, 6, 6, 7},  // positions_major
                    {1, 2, 3, 5, 0, 0, 1, 0},  // positions_minor
                },
                MockSample{
                    0,              // seq_id
                    {6, 6, 7, 7},   // positions_major
                    {0, 1, 0, 1},   // positions_minor
                },
            },
            std::nullopt,   // region
            false,          // expect_throw
            {   // Expected results.
                TrimInfo{0, 9, false},
                TrimInfo{1, 6, false},
                TrimInfo{1, 4, false},
            },
        },
        TestCase{
            "Medaka test. Two samples with a gap in between.",
            {   // Input samples.
                MockSample{
                    0,                                  // seq_id
                    {0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 4},  // positions_major
                    {0, 1, 0, 0, 1, 2, 0, 0, 1, 2, 3},  // positions_minor
                },
                MockSample{
                    0,              // seq_id
                    {6, 6, 7, 7},   // positions_major
                    {0, 1, 0, 1},   // positions_minor
                },
            },
            std::nullopt,   // region
            false,          // expect_throw
            {   // Expected results.
                TrimInfo{0, 11, false},
                TrimInfo{0, 4, false},
            },
        },
        TestCase{
            "Medaka test. Input samples are out of order and overlapping. Should throw.",
            {   // Input samples.
                MockSample{
                    0,              // seq_id
                    {6, 6, 7, 7},   // positions_major
                    {0, 1, 0, 1},   // positions_minor
                },
                MockSample{
                    0,                         // seq_id
                    {4, 4, 4, 4, 5, 6, 6, 7},  // positions_major
                    {1, 2, 3, 5, 0, 0, 1, 0},  // positions_minor
                },
                MockSample{
                    0,                                  // seq_id
                    {0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 4},  // positions_major
                    {0, 1, 0, 0, 1, 2, 0, 0, 1, 2, 3},  // positions_minor
                },
            },
            std::nullopt,   // region
            true,           // expect_throw
            {   // Expected results.
            },
        },
        TestCase{
            "Input samples are out of order but NOT overlapping. Should throw because only forward oriented relationship is allowed.",
            {   // Input samples.
                MockSample{
                    0,              // seq_id
                    {6, 6, 7, 7},   // positions_major
                    {0, 1, 0, 1},   // positions_minor
                },
                MockSample{
                    0,                                  // seq_id
                    {0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 4},  // positions_major
                    {0, 1, 0, 0, 1, 2, 0, 0, 1, 2, 3},  // positions_minor
                },
            },
            std::nullopt,   // region
            true,           // expect_throw
            {   // Expected results.
            },
        },
        TestCase{
            "Three samples on different reference. No trimming should be applied.",
            {   // Input samples.
                MockSample{
                    0,                                  // seq_id
                    {0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 4},  // positions_major
                    {0, 1, 0, 0, 1, 2, 0, 0, 1, 2, 3},  // positions_minor
                },
                MockSample{
                    1,                         // seq_id
                    {4, 4, 4, 4, 5, 6, 6, 7},  // positions_major
                    {1, 2, 3, 5, 0, 0, 1, 0},  // positions_minor
                },
                MockSample{
                    2,              // seq_id
                    {6, 6, 7, 7},   // positions_major
                    {0, 1, 0, 1},   // positions_minor
                },
            },
            std::nullopt,   // region
            false,          // expect_throw
            {   // Expected results.
                TrimInfo{0, 11, false},
                TrimInfo{0, 8, false},
                TrimInfo{0, 4, false},
            },
        },
        TestCase{
            " Three samples per reference. Should trim.",
            {   // Input samples.
                MockSample{
                    0,                                  // seq_id
                    {0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 4},  // positions_major
                    {0, 1, 0, 0, 1, 2, 0, 0, 1, 2, 3},  // positions_minor
                },
                MockSample{
                    0,                         // seq_id
                    {4, 4, 4, 4, 5, 6, 6, 7},  // positions_major
                    {1, 2, 3, 5, 0, 0, 1, 0},  // positions_minor
                },
                MockSample{
                    0,              // seq_id
                    {6, 6, 7, 7},   // positions_major
                    {0, 1, 0, 1},   // positions_minor
                },
                MockSample{
                    1,                                  // seq_id
                    {0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 4},  // positions_major
                    {0, 1, 0, 0, 1, 2, 0, 0, 1, 2, 3},  // positions_minor
                },
                MockSample{
                    1,                         // seq_id
                    {4, 4, 4, 4, 5, 6, 6, 7},  // positions_major
                    {1, 2, 3, 5, 0, 0, 1, 0},  // positions_minor
                },
                MockSample{
                    1,              // seq_id
                    {6, 6, 7, 7},   // positions_major
                    {0, 1, 0, 1},   // positions_minor
                },
            },
            std::nullopt,   // region
            false,          // expect_throw
            {   // Expected results.
                TrimInfo{0, 9, false},
                TrimInfo{1, 6, false},
                TrimInfo{1, 4, false},
                TrimInfo{0, 9, false},
                TrimInfo{1, 6, false},
                TrimInfo{1, 4, false},
            },
        },
        TestCase{
            "Use region, but the region is larger than what samples span. Region trimming won't influence the results.",
            {   // Input samples.
                MockSample{
                    0,                                  // seq_id
                    {0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 4},  // positions_major
                    {0, 1, 0, 0, 1, 2, 0, 0, 1, 2, 3},  // positions_minor
                },
                MockSample{
                    0,                         // seq_id
                    {4, 4, 4, 4, 5, 6, 6, 7},  // positions_major
                    {1, 2, 3, 5, 0, 0, 1, 0},  // positions_minor
                },
                MockSample{
                    0,              // seq_id
                    {6, 6, 7, 7},   // positions_major
                    {0, 1, 0, 1},   // positions_minor
                },
            },
            RegionInt{0, 0, 1000},   // region
            false,          // expect_throw
            {   // Expected results.
                TrimInfo{0, 9, false},
                TrimInfo{1, 6, false},
                TrimInfo{1, 4, false},
            },
        },
        TestCase{
            "Trim to region which is shorter than the span of windows. Three samples overlapping, and one on a different reference. Trimming should be applied between samples, and on region start/end.",
            {   // Input samples.
                MockSample{
                    0,                                  // seq_id
                    {0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 4},  // positions_major
                    {0, 1, 0, 0, 1, 2, 0, 0, 1, 2, 3},  // positions_minor
                },
                MockSample{
                    0,                         // seq_id
                    {4, 4, 4, 4, 5, 6, 6, 7},  // positions_major
                    {1, 2, 3, 5, 0, 0, 1, 0},  // positions_minor
                },
                MockSample{
                    0,              // seq_id
                    {6, 6, 7, 7},   // positions_major
                    {0, 1, 0, 1},   // positions_minor
                },
                MockSample{
                    1,                                  // seq_id
                    {0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 4},  // positions_major
                    {0, 1, 0, 0, 1, 2, 0, 0, 1, 2, 3},  // positions_minor
                },
            },
            RegionInt{0, 1, 5},   // region
            false,          // expect_throw
            {   // Expected results.
                TrimInfo{2, 9, false},
                TrimInfo{1, 4, false},
                TrimInfo{-1, -1, false},    // Filtered out.
                TrimInfo{-1, -1, false},    // Filtered out.
            },
        },
        TestCase{
            "Use region, but region start is not valid. Should throw.",
            {   // Input samples.
                MockSample{
                    0,                                  // seq_id
                    {0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 4},  // positions_major
                    {0, 1, 0, 0, 1, 2, 0, 0, 1, 2, 3},  // positions_minor
                },
            },
            RegionInt{0, -1, 1000},   // region
            true,          // expect_throw
            {   // Expected results.
            },
        },
        TestCase{
            "Use region, but region end is not valid. Should throw.",
            {   // Input samples.
                MockSample{
                    0,                                  // seq_id
                    {0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 4},  // positions_major
                    {0, 1, 0, 0, 1, 2, 0, 0, 1, 2, 3},  // positions_minor
                },
            },
            RegionInt{0, 1000, -1},   // region
            true,          // expect_throw
            {   // Expected results.
            },
        },
        TestCase{
            "Use region, but region seq_id is not valid. Should throw.",
            {   // Input samples.
                MockSample{
                    0,                                  // seq_id
                    {0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 4},  // positions_major
                    {0, 1, 0, 0, 1, 2, 0, 0, 1, 2, 3},  // positions_minor
                },
            },
            RegionInt{-1, 1000, 2000},   // region
            true,          // expect_throw
            {   // Expected results.
            },
        },
    }));
    // clang-format on

    INFO(TEST_GROUP << " Test name: " << test_case.name);
    std::vector<Sample> samples;
    for (const auto& mock_sample : test_case.samples) {
        Sample new_sample;
        new_sample.seq_id = mock_sample.seq_id;
        new_sample.positions_major = mock_sample.positions_major;
        new_sample.positions_minor = mock_sample.positions_minor;
        samples.emplace_back(std::move(new_sample));
    }

    if (test_case.expect_throw) {
        CHECK_THROWS(trim_samples(samples, test_case.region));
    } else {
        const std::vector<TrimInfo> result = trim_samples(samples, test_case.region);
        CHECK(test_case.expected == result);
    }
}

}  // namespace