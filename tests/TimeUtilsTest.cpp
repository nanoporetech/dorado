#include "utils/time_utils.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <chrono>
#include <limits>

using std::make_tuple;
using namespace std::chrono;
using namespace std::chrono_literals;

#define CUT_TAG "[TimeUtils]"

CATCH_TEST_CASE(CUT_TAG ": std::string <-> time_t", CUT_TAG) {
    using sys = std::chrono::system_clock;
    using time_point = sys::time_point;

    CATCH_SECTION("With timezone HH:MM") {
        auto [timestamp, unix_time_ms] = GENERATE(table<std::string, time_t>({
                make_tuple("1970-01-01T00:00:00.000000+00:00",
                           sys::to_time_t(time_point(0s)) * 1000),
                make_tuple("1970-01-02T00:00:00.000000+00:00",
                           sys::to_time_t(time_point(24h)) * 1000),
                make_tuple("1971-01-02T00:00:00.000000+00:00",
                           sys::to_time_t(time_point(8784h)) * 1000),
                make_tuple("1975-01-02T00:00:00.000000+00:00",
                           sys::to_time_t(time_point(43848h)) * 1000),
                make_tuple("1975-01-02T00:00:00.456000+00:00",
                           sys::to_time_t(time_point(43848h)) * 1000 + 456),
        }));
        CATCH_CAPTURE(timestamp);

        auto result_str = dorado::utils::get_string_timestamp_from_unix_time_ms(unix_time_ms);
        CATCH_CHECK(result_str == timestamp);

        auto result_ms = dorado::utils::get_unix_time_ms_from_string_timestamp(timestamp);
        CATCH_CHECK(result_ms == unix_time_ms);
    }

    CATCH_SECTION("With timezone Z") {
        auto [timestamp, unix_time_ms] = GENERATE(table<std::string, time_t>({
                make_tuple("1970-01-01T00:00:00Z", sys::to_time_t(time_point(0s)) * 1000),
                make_tuple("1970-01-02T00:00:00Z", sys::to_time_t(time_point(24h)) * 1000),
                make_tuple("1971-01-02T00:00:00Z", sys::to_time_t(time_point(8784h)) * 1000),
                make_tuple("1975-01-02T00:00:00Z", sys::to_time_t(time_point(43848h)) * 1000),
        }));
        CATCH_CAPTURE(timestamp);
        auto result_ms = dorado::utils::get_unix_time_ms_from_string_timestamp(timestamp);
        CATCH_CHECK(result_ms == unix_time_ms);
    }

    CATCH_SECTION("With microseconds") {
        auto [timestamp, unix_time_ms] = GENERATE(table<std::string, time_t>({
                make_tuple("1970-01-01T00:00:00.000000+00:00",
                           sys::to_time_t(time_point(0s)) * 1000),
                make_tuple("1970-01-02T00:00:00.000101+00:00",
                           sys::to_time_t(time_point(24h)) * 1000),
                make_tuple("1971-01-02T00:00:00.456000+00:00",
                           sys::to_time_t(time_point(8784h)) * 1000 + 456),
                make_tuple("1975-01-02T00:00:00.456123+00:00",
                           sys::to_time_t(time_point(43848h)) * 1000 + 456),
        }));
        CATCH_CAPTURE(timestamp);
        auto result_ms = dorado::utils::get_unix_time_ms_from_string_timestamp(timestamp);
        CATCH_CHECK(result_ms == unix_time_ms);
    }

    CATCH_SECTION("Seconds + Z") {
        auto [timestamp, unix_time_s] = GENERATE(table<std::string, time_t>({
                make_tuple("1970-01-01T00:00:00Z", sys::to_time_t(time_point(0s))),
                make_tuple("1970-01-02T00:00:01Z", sys::to_time_t(time_point(24h)) + 1),
                make_tuple("1971-01-02T00:02:00Z", sys::to_time_t(time_point(8784h)) + 2 * 60),
                make_tuple("1975-01-02T03:00:00Z", sys::to_time_t(time_point(43848h)) + 3 * 3600),
        }));
        CATCH_CAPTURE(timestamp);
        auto result_str = dorado::utils::get_string_timestamp_from_unix_time_sZ(unix_time_s);
        CATCH_CHECK(result_str == timestamp);
    }
}

CATCH_TEST_CASE(CUT_TAG ": MinKnow datetime format from pod5 timestamp", CUT_TAG) {
    CATCH_CHECK(dorado::utils::get_minknow_timestamp_from_unix_time_ms(0) == "19700101_0000");
    CATCH_CHECK(dorado::utils::get_minknow_timestamp_from_unix_time_ms(1110371400000) ==
                "20050309_1230");
    CATCH_CHECK(dorado::utils::get_minknow_timestamp_from_unix_time_ms(1750411401000) ==
                "20250620_0923");
}
