#include "utils/stl_utils.h"

#include <catch2/catch_test_macros.hpp>

#include <map>
#include <string>
#include <unordered_map>

#define TEST_GROUP "[dorado::utils::stl_utils]"

namespace dorado::utils::stl_utils_test {

CATCH_TEST_CASE("get_value_or_default returns the mapped value when the key is present",
                TEST_GROUP) {
    const std::map<std::string, std::string> map{{"a", "1"}, {"b", "2"}};
    CATCH_CHECK(get_value_or_default(map, "a", std::string("default")) == "1");
    CATCH_CHECK(get_value_or_default(map, "b", std::string("default")) == "2");
}

CATCH_TEST_CASE("get_value_or_default returns the default when the key is absent", TEST_GROUP) {
    const std::map<std::string, std::string> map{{"a", "1"}};
    CATCH_CHECK(get_value_or_default(map, "missing", std::string("default")) == "default");
}

CATCH_TEST_CASE("get_value_or_default returns the default for an empty map", TEST_GROUP) {
    const std::unordered_map<std::string, std::string> map;
    CATCH_CHECK(get_value_or_default(map, "anything", std::string("fallback")) == "fallback");
}

CATCH_TEST_CASE("get_value_or_default accepts a const char* default for a std::string mapped_type",
                TEST_GROUP) {
    // `Map::mapped_type` is a non-deduced context, so a string literal converts to std::string
    // instead of forcing the default's type onto the return. The result must be a std::string.
    const std::map<std::string, std::string> map{{"a", "1"}};
    const std::string present = get_value_or_default(map, "a", "literal_default");
    const std::string absent = get_value_or_default(map, "missing", "literal_default");
    CATCH_CHECK(present == "1");
    CATCH_CHECK(absent == "literal_default");
}

CATCH_TEST_CASE("get_value_or_default works with non-string key and value types", TEST_GROUP) {
    const std::map<int, int> map{{1, 10}, {2, 20}};
    CATCH_CHECK(get_value_or_default(map, 1, -1) == 10);
    CATCH_CHECK(get_value_or_default(map, 2, -1) == 20);
    CATCH_CHECK(get_value_or_default(map, 3, -1) == -1);
}

CATCH_TEST_CASE("get_value_or_default works with std::unordered_map", TEST_GROUP) {
    const std::unordered_map<std::string, int> map{{"x", 7}};
    CATCH_CHECK(get_value_or_default(map, "x", 0) == 7);
    CATCH_CHECK(get_value_or_default(map, "y", 0) == 0);
}

CATCH_TEST_CASE("get_value_or_default returns the mapped value even when it equals the default",
                TEST_GROUP) {
    const std::map<std::string, std::string> map{{"a", "same"}};
    CATCH_CHECK(get_value_or_default(map, "a", std::string("same")) == "same");
}

CATCH_TEST_CASE("get_value_or_default does not insert missing keys", TEST_GROUP) {
    // Unlike operator[], a lookup miss must not mutate the map.
    std::map<std::string, std::string> map{{"a", "1"}};
    const auto size_before = map.size();
    get_value_or_default(map, "missing", std::string("default"));
    CATCH_CHECK(map.size() == size_before);
    CATCH_CHECK(map.count("missing") == 0);
}

}  // namespace dorado::utils::stl_utils_test
