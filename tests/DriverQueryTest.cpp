#include "utils/driver_query.h"

#include <catch2/catch.hpp>

#include <algorithm>
#include <cctype>

#define DEFINE_TEST(name) TEST_CASE("DriverQueryTest: " name, "[DriverQueryTest]")

namespace {

DEFINE_TEST("Driver available if torch says we have CUDA") {
    auto driver_version = dorado::utils::get_nvidia_driver_version();
    if (torch::hasCUDA()) {
        REQUIRE(driver_version.has_value());
    }
}

DEFINE_TEST("Valid version string") {
    auto driver_version = dorado::utils::get_nvidia_driver_version();
    if (driver_version.has_value()) {
        CHECK(!driver_version->empty());
        // Version string should be made up of digits and dots only.
        auto is_valid_char = [](char c) {
            return std::isdigit(static_cast<unsigned char>(c)) || c == '.';
        };
        CHECK(std::all_of(driver_version->begin(), driver_version->end(), is_valid_char));
        CHECK(std::count(driver_version->begin(), driver_version->end(), '.') <= 3);
    }
}

DEFINE_TEST("Multiple calls return the same result") {
    auto driver_version_0 = dorado::utils::get_nvidia_driver_version();
    auto driver_version_1 = dorado::utils::get_nvidia_driver_version();
    CHECK(driver_version_0 == driver_version_1);
}

#if defined(__APPLE__)
DEFINE_TEST("No driver on Apple") {
    auto driver_version = dorado::utils::get_nvidia_driver_version();
    CHECK(!driver_version.has_value());
}
#endif  // __APPLE__

}  // namespace
