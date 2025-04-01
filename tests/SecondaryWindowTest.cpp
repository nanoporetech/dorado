#include "secondary/consensus/window.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cstdint>
#include <vector>

namespace dorado::secondary::sample::tests {

#define TEST_GROUP "[SecondaryConsensus]"

CATCH_TEST_CASE("create_windows tests", TEST_GROUP) {
    struct TestCase {
        std::string name;
        int32_t seq_id = 0;
        int64_t seq_start = 0;
        int64_t seq_end = 0;
        int64_t seq_len = 0;
        int32_t window_len = 0;
        int32_t window_overlap = 0;
        std::vector<Window> expected;
    };

    // clang-format off
    auto [test_case] = GENERATE(table<TestCase>({
        TestCase{
            "Coordinates and lengths all zero", 0, 0, 0, 0, 0, 0,
            {},
        },
        TestCase{
            "Normal full-contig", 0, 0, 10000, 10000, 2000, 100,
            {
                Window{0, 10000, 0, 2000, 0, 2000},
                Window{0, 10000, 1900, 3900, 2000, 3900},
                Window{0, 10000, 3800, 5800, 3900, 5800},
                Window{0, 10000, 5700, 7700, 5800, 7700},
                Window{0, 10000, 7600, 9600, 7700, 9600},
                Window{0, 10000, 9500, 10000, 9600, 10000},
            },
        },
        TestCase{
            "Normal short", 0, 0, 500, 10000, 2000, 100,
            {
                Window{0, 10000, 0, 500, 0, 500},
            },
        },
        TestCase{
            "Normal, internal region", 0, 100, 3000, 10000, 2000, 100,
            {
                Window{0, 10000, 100, 2100, 100, 2100},
                Window{0, 10000, 2000, 3000, 2100, 3000},
            },
        },
        TestCase{
            "Zero-length input interval. Returns empty.", 0, 5, 5, 10000, 2000, 100,
            {},
        },
        TestCase{
            "End coordinate over contig length. Returns empty.", 0, 0, 20000, 10000, 2000, 100,
            {},
        },
        TestCase{
            "Start coordinate is < 0. Returns empty.", 0, -5, 10000, 10000, 2000, 100,
            {},
        },
        TestCase{
            "Sequence length is < 0. Returns empty.", 0, 0, 10000, -10000, 2000, 100,
            {},
        },
        TestCase{
            "Start coordinate is > end coordinate. Returns empty.", 0, 12000, 10000, 10000, 2000, 100,
            {},
        },
        TestCase{
            "Window overlap is >= window_len / 2. Returns empty.", 0, 0, 10000, 10000, 2000, 1000,
            {},
        },
        TestCase{
            "Window len == 0. Returns empty.", 0, 0, 10000, 10000, 0, 1000,
            {},
        },
        TestCase{
            "Window len < 0. Returns empty.", 0, 0, 10000, 10000, -1, 1000,
            {},
        },
        TestCase{
            "Window overlap < 0. Returns empty.", 0, 0, 10000, 10000, 2000, -1,
            {},
        },
        TestCase{
            "Window overlap == 0. This is fine, should return non-overlapping windows.", 0, 0, 10000, 10000, 2000, 0,
            {
                Window{0, 10000, 0, 2000, 0, 2000},
                Window{0, 10000, 2000, 4000, 2000, 4000},
                Window{0, 10000, 4000, 6000, 4000, 6000},
                Window{0, 10000, 6000, 8000, 6000, 8000},
                Window{0, 10000, 8000, 10000, 8000, 10000},
            },
        },
    }));
    // clang-format on

    CATCH_INFO(TEST_GROUP << " Test name: " << test_case.name);
    const std::vector<Window> result =
            create_windows(test_case.seq_id, test_case.seq_start, test_case.seq_end,
                           test_case.seq_len, test_case.window_len, test_case.window_overlap);
    CATCH_CHECK(result == test_case.expected);
}

}  // namespace dorado::secondary::sample::tests