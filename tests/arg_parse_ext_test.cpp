#include "utils/arg_parse_ext.h"

#include <catch2/catch.hpp>

#define CUT_TAG "[dorado::utils::arg_parse]"

namespace dorado::utils::arg_parse::test {

TEST_CASE(CUT_TAG " Check number string to long conversion", CUT_TAG) {
    SECTION("convert #K") { CHECK(parse_string_to_size("5K") == 5000); }
    SECTION("convert #M") { CHECK(parse_string_to_size("5.3M") == 5300000); }
    SECTION("convert #G") { CHECK(parse_string_to_size("5G") == 5000000000); }
    SECTION("convert #") { CHECK(parse_string_to_size("50") == 50); }
    SECTION("convert 0") { CHECK(parse_string_to_size("000") == 0); }
    SECTION("convert empty") { CHECK_THROWS(parse_string_to_size("")); }
    SECTION("convert unexpected size character") { CHECK_THROWS(parse_string_to_size("5L")); }
    SECTION("convert not a number") { CHECK_THROWS(parse_string_to_size("abcd")); }
}

TEST_CASE(CUT_TAG " Check numbers string to vector of longs conversion", CUT_TAG) {
    SECTION("convert #") { CHECK(parse_string_to_sizes("5K").size() == 1); }
    SECTION("convert #,#") { CHECK(parse_string_to_sizes("5.3M,5G").size() == 2); }
    SECTION("convert #,#,#,#") { CHECK(parse_string_to_sizes("5.3M,5G,50,000").size() == 4); }
    SECTION("convert empty") { CHECK_THROWS(parse_string_to_sizes("")); }
    SECTION("convert separator") { CHECK_THROWS(parse_string_to_sizes(",")); }
    SECTION("convert unexpected size character") { CHECK_THROWS(parse_string_to_sizes("5L,1")); }
    SECTION("convert not a number") { CHECK_THROWS(parse_string_to_sizes("1,abcd")); }
}

}  // namespace dorado::utils::arg_parse::test