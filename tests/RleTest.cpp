#include "utils/rle.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#define TEST_GROUP "[RLE]"

namespace dorado::rle::tests {

CATCH_TEST_CASE("run_length_encode handles empty input", TEST_GROUP) {
    const std::vector<int32_t> data;
    const auto result = run_length_encode(data);
    CATCH_CHECK(std::empty(result));
}

CATCH_TEST_CASE("run_length_encode handles single element", TEST_GROUP) {
    const std::vector<int32_t> data{42};
    const auto result = run_length_encode(data);
    CATCH_REQUIRE(result.size() == 1);
    CATCH_CHECK(result[0] == std::make_tuple(0, 1, 42));
}

CATCH_TEST_CASE("run_length_encode handles uniform data", TEST_GROUP) {
    std::vector<int32_t> data{5, 5, 5, 5};
    auto result = run_length_encode(data);
    CATCH_REQUIRE(std::size(result) == 1);
    CATCH_CHECK(result[0] == std::make_tuple(0, 4, 5));
}

CATCH_TEST_CASE("run_length_encode handles non-uniform data", TEST_GROUP) {
    const std::vector<int32_t> data{1, 1, 2, 2, 2, 3, 3, 4};
    const auto result = run_length_encode(data);
    CATCH_REQUIRE(std::size(result) == 4);
    CATCH_CHECK(result[0] == std::make_tuple(0, 2, 1));
    CATCH_CHECK(result[1] == std::make_tuple(2, 5, 2));
    CATCH_CHECK(result[2] == std::make_tuple(5, 7, 3));
    CATCH_CHECK(result[3] == std::make_tuple(7, 8, 4));
}

CATCH_TEST_CASE("run_length_encode handles custom comparator", TEST_GROUP) {
    struct TestStruct {
        int32_t val = 0;
        std::string str;
        bool operator==(const TestStruct& other) const {
            return val == other.val && str == other.str;
        }
    };
    const std::vector<TestStruct> data{
            {1, "a"}, {2, "a"}, {3, "a"}, {4, "b"}, {5, "c"},
    };
    const auto result = run_length_encode(
            data, [](const TestStruct& a, const TestStruct& b) { return a.str == b.str; });
    CATCH_REQUIRE(std::size(result) == 3);
    CATCH_CHECK(result[0] == std::make_tuple(0, 3, TestStruct{1, "a"}));
    CATCH_CHECK(result[1] == std::make_tuple(3, 4, TestStruct{4, "b"}));
    CATCH_CHECK(result[2] == std::make_tuple(4, 5, TestStruct{5, "c"}));
}

CATCH_TEST_CASE("run_length_encode handles strings", TEST_GROUP) {
    std::vector<std::string> data{"cat", "cat", "dog", "dog", "mouse"};
    auto result = run_length_encode(data);
    CATCH_REQUIRE(std::size(result) == 3);
    CATCH_CHECK(result[0] == std::make_tuple(0, 2, "cat"));
    CATCH_CHECK(result[1] == std::make_tuple(2, 4, "dog"));
    CATCH_CHECK(result[2] == std::make_tuple(4, 5, "mouse"));
}

CATCH_TEST_CASE("run_length_encode handles unsorted data", TEST_GROUP) {
    const std::vector<int32_t> data{1, 2, 2, 1, 1, 1, 3, 3, 3};
    const auto result = run_length_encode(data);
    CATCH_REQUIRE(std::size(result) == 4);
    CATCH_CHECK(result[0] == std::make_tuple(0, 1, 1));
    CATCH_CHECK(result[1] == std::make_tuple(1, 3, 2));
    CATCH_CHECK(result[2] == std::make_tuple(3, 6, 1));
    CATCH_CHECK(result[3] == std::make_tuple(6, 9, 3));
}

}  // namespace dorado::rle::tests