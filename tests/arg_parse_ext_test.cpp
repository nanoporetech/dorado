#include "utils/arg_parse_ext.h"

#include <catch2/catch_test_macros.hpp>

#define CUT_TAG "[dorado::utils::arg_parse]"

namespace dorado::utils::arg_parse::test {

CATCH_TEST_CASE(CUT_TAG " Check number string to long conversion", CUT_TAG) {
    CATCH_SECTION("convert #K") { CATCH_CHECK(parse_string_to_size("5K") == 5000); }
    CATCH_SECTION("convert #M") { CATCH_CHECK(parse_string_to_size("5.3M") == 5300000); }
    CATCH_SECTION("convert #G") { CATCH_CHECK(parse_string_to_size("5G") == 5000000000); }
    CATCH_SECTION("convert #") { CATCH_CHECK(parse_string_to_size("50") == 50); }
    CATCH_SECTION("convert 0") { CATCH_CHECK(parse_string_to_size("000") == 0); }
    CATCH_SECTION("convert empty") { CATCH_CHECK_THROWS(parse_string_to_size("")); }
    CATCH_SECTION("convert unexpected size character") {
        CATCH_CHECK_THROWS(parse_string_to_size("5L"));
    }
    CATCH_SECTION("convert not a number") { CATCH_CHECK_THROWS(parse_string_to_size("abcd")); }
}

CATCH_TEST_CASE(CUT_TAG " Check numbers string to vector of longs conversion", CUT_TAG) {
    CATCH_SECTION("convert #") { CATCH_CHECK(parse_string_to_sizes("5K").size() == 1); }
    CATCH_SECTION("convert #,#") { CATCH_CHECK(parse_string_to_sizes("5.3M,5G").size() == 2); }
    CATCH_SECTION("convert #,#,#,#") {
        CATCH_CHECK(parse_string_to_sizes("5.3M,5G,50,000").size() == 4);
    }
    CATCH_SECTION("convert empty") { CATCH_CHECK_THROWS(parse_string_to_sizes("")); }
    CATCH_SECTION("convert separator") { CATCH_CHECK_THROWS(parse_string_to_sizes(",")); }
    CATCH_SECTION("convert unexpected size character") {
        CATCH_CHECK_THROWS(parse_string_to_sizes("5L,1"));
    }
    CATCH_SECTION("convert not a number") { CATCH_CHECK_THROWS(parse_string_to_sizes("1,abcd")); }
}

}  // namespace dorado::utils::arg_parse::test