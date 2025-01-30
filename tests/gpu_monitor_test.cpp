#include "torch_utils/gpu_monitor.h"

#include <catch2/catch_test_macros.hpp>
#include <torch/torch.h>

#include <algorithm>
#include <cctype>
#include <sstream>

#define CUT_TAG "[dorado::utils::gpu_monitor]"
#define DEFINE_TEST(name) CATCH_TEST_CASE(CUT_TAG " " name, CUT_TAG)
#define DEFINE_TEST_FIXTURE_METHOD(name) \
    CATCH_TEST_CASE_METHOD(GpuMonitorTestFixture, CUT_TAG " " name, CUT_TAG)

namespace {

class GpuMonitorTestFixture {
protected:
    unsigned int num_devices;
    std::optional<unsigned int> first_accessible_device;

public:
    GpuMonitorTestFixture() {
        num_devices = dorado::utils::gpu_monitor::get_device_count();
        for (unsigned int index{}; index < num_devices; ++index) {
            if (dorado::utils::gpu_monitor::detail::is_accessible_device(index)) {
                first_accessible_device = index;
                break;
            }
        }
    }
};
}  // namespace

namespace dorado::utils::gpu_monitor::test {

DEFINE_TEST("get_nvidia_driver_version has value if torch::hasCUDA") {
    auto driver_version = get_nvidia_driver_version();
    if (torch::hasCUDA()) {
        CATCH_REQUIRE(driver_version.has_value());
    }
}

DEFINE_TEST("get_nvidia_driver_version retruns valid version string") {
    auto driver_version = get_nvidia_driver_version();
    if (driver_version.has_value()) {
        CATCH_CHECK(!driver_version->empty());
        // Version string should be made up of digits and dots only.
        auto is_valid_char = [](char c) {
            return std::isdigit(static_cast<unsigned char>(c)) || c == '.';
        };
        CATCH_CHECK(std::all_of(driver_version->begin(), driver_version->end(), is_valid_char));
        CATCH_CHECK(std::count(driver_version->begin(), driver_version->end(), '.') <= 3);
    }
}

DEFINE_TEST("get_nvidia_driver_version multiple calls return the same result") {
    auto driver_version_0 = get_nvidia_driver_version();
    auto driver_version_1 = get_nvidia_driver_version();
    CATCH_CHECK(driver_version_0.has_value() == driver_version_1.has_value());
    if (driver_version_0.has_value()) {
        CATCH_REQUIRE(*driver_version_0 == *driver_version_1);
    }
}

#if defined(__APPLE__)
DEFINE_TEST("get_nvidia_driver_version does not have value on Apple") {
    auto driver_version = get_nvidia_driver_version();
    CATCH_CHECK(!driver_version.has_value());
}
#endif  // __APPLE__

#if DORADO_TX2
DEFINE_TEST("get_nvidia_driver_version always has a value on Jetson") {
    auto driver_version = get_nvidia_driver_version();
    CATCH_CHECK(driver_version.has_value());
}
#endif  // DORADO_TX2

DEFINE_TEST("parse_nvidia_version_line parameterised test") {
    const struct {
        std::string_view test_name;
        std::string_view line;
        bool valid;
        std::string_view version;
    } tests[]{
            {
                    "Valid version line",
                    "NVRM version: NVIDIA UNIX x86_64 Kernel Module  520.61.05  Thu Sep 29 "
                    "05:30:25 UTC 2022",
                    true,
                    "520.61.05",
            },
            {
                    "Compiler line is ignored",
                    "GCC version:  gcc version 9.4.0 (Ubuntu 9.4.0-1ubuntu1~20.04.1)",
                    false,
                    "",
            },
            {
                    "Valid version line from a different machine",
                    "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  378.13  Release "
                    "Build  (builder)",
                    true,
                    "378.13",
            },
            {
                    "Missing <info> and patch version",
                    "NVRM version: module name  123.456",
                    true,
                    "123.456",
            },
            {
                    "TX2 line",
                    "NVRM version: NVIDIA UNIX Kernel Module for aarch64  34.1.1  Release Build  "
                    "(buildbrain@mobile-u64-5414-d7000)  Mon May 16 21:12:24 PDT 2022",
                    true,
                    "34.1.1",
            },
    };

    for (const auto &test : tests) {
        CATCH_CAPTURE(test.test_name);
        auto version = detail::parse_nvidia_version_line(test.line);
        CATCH_CHECK(version.has_value() == test.valid);
        if (version.has_value() && test.valid) {
            CATCH_CHECK(version.value() == test.version);
        }
    }
}

DEFINE_TEST("parse_nvidia_tegra_line parameterised test") {
    const struct {
        std::string_view test_name;
        std::string line;
        bool valid;
        std::string_view version;
    } tests[]{
            {
                    "Valid version line",
                    "# R32 (release), REVISION: 4.3, GCID: 21589087, BOARD: t186ref, EABI: "
                    "aarch64, DATE: Fri Jun 26 04:34:27 UTC 2020",
                    true,
                    "32.4.3",
            },
            {
                    "Extraneous parts are ignored",
                    "# R123 (release), REVISION: 456.789",
                    true,
                    "123.456.789",
            },
            {
                    "Invalid line",
                    "This is not the data you're looking for",
                    false,
                    "",
            },
            {
                    "Valid lines from a different machine",
                    "# R28 (release), REVISION: 2.0, GCID: 10567845, BOARD: t186ref, EABI: "
                    "aarch64, DATE: Fri Mar  2 04:57:01 UTC 2018\n",
                    true,
                    "28.2.0",
            },
    };

    for (const auto &test : tests) {
        CATCH_CAPTURE(test.test_name);
        auto version = detail::parse_nvidia_tegra_line(test.line);
        CATCH_CHECK(version.has_value() == test.valid);
        if (version.has_value() && test.valid) {
            CATCH_CHECK(version.value() == test.version);
        }
    }
}

DEFINE_TEST("get_device_count does not throw") { CATCH_REQUIRE_NOTHROW(get_device_count()); }

DEFINE_TEST_FIXTURE_METHOD("get_device_status_info with valid device does not throw") {
    if (!first_accessible_device) {
        return;
    }
    CATCH_CHECK_NOTHROW(get_device_status_info(*first_accessible_device));
}

DEFINE_TEST_FIXTURE_METHOD("get_device_status_info with valid device has assigned value") {
    if (!first_accessible_device) {
        return;
    }
    auto info = get_device_status_info(*first_accessible_device);
    CATCH_CHECK(info.has_value());
}

DEFINE_TEST_FIXTURE_METHOD(
        "get_device_status_info with valid device returns with correct device_id") {
    if (!first_accessible_device) {
        return;
    }
    auto info = get_device_status_info(*first_accessible_device);

    // N.B. test may fail has_value() check if a CI runner GPU does not support an nvml query
    // in which case consider rewriting the specific test to pass if the optional is not set
    CATCH_REQUIRE(info.has_value());
    CATCH_CHECK(info->device_index == *first_accessible_device);
}

DEFINE_TEST_FIXTURE_METHOD(
        "get_device_status_info with valid device returns non-zero temperature") {
    if (!first_accessible_device) {
        return;
    }
    auto info = get_device_status_info(*first_accessible_device);

    // N.B. test may fail has_value() check if a CI runner GPU does not support an nvml query
    // in which case consider rewriting the specific test to pass if the optional is not set
    CATCH_REQUIRE(info.has_value());
    CATCH_CAPTURE(info->current_temperature_error);
    CATCH_REQUIRE(info->current_temperature.has_value());
    CATCH_CHECK(*info->current_temperature > 0);
}

DEFINE_TEST_FIXTURE_METHOD(
        "get_device_status_info with valid device returns non-zero shutdown threshold") {
    if (!first_accessible_device) {
        return;
    }
    auto info = get_device_status_info(*first_accessible_device);

    // N.B. test may fail has_value() check if a CI runner GPU does not support an nvml query
    // in which case consider rewriting the specific test to pass if the optional is not set
    CATCH_REQUIRE(info.has_value());
    CATCH_CAPTURE(info->gpu_shutdown_temperature_error);
    CATCH_REQUIRE(info->gpu_shutdown_temperature.has_value());
    CATCH_CHECK(*info->gpu_shutdown_temperature > 0);
}

DEFINE_TEST_FIXTURE_METHOD(
        "get_device_status_info with valid device returns non-zero slowdown threshold") {
    if (!first_accessible_device) {
        return;
    }
    auto info = get_device_status_info(*first_accessible_device);

    // N.B. test may fail has_value() check if a CI runner GPU does not support an nvml query
    // in which case consider rewriting the specific test to pass if the optional is not set
    CATCH_REQUIRE(info.has_value());
    CATCH_CAPTURE(info->gpu_slowdown_temperature_error);
    CATCH_REQUIRE(info->gpu_slowdown_temperature.has_value());
    CATCH_CHECK(*info->gpu_slowdown_temperature > 0);
}

DEFINE_TEST_FIXTURE_METHOD(
        "get_device_status_info with valid device returns with non-zero max operating "
        "temperature") {
    if (!first_accessible_device) {
        return;
    }
    auto info = get_device_status_info(*first_accessible_device);

    // N.B. test may fail has_value() check if a CI runner GPU does not support an nvml query
    // in which case consider rewriting the specific test to pass if the optional is not set
    CATCH_REQUIRE(info.has_value());
    CATCH_CAPTURE(info->gpu_max_operating_temperature_error);
    CATCH_REQUIRE(info->gpu_max_operating_temperature.has_value());
    CATCH_CHECK(*info->gpu_max_operating_temperature > 0);
}

DEFINE_TEST_FIXTURE_METHOD(
        "get_device_status_info with valid device returns with non-zero current_power_usage") {
    if (!first_accessible_device) {
        return;
    }
    auto info = get_device_status_info(*first_accessible_device);

    // N.B. test may fail has_value() check if a CI runner GPU does not support an nvml query
    // in which case consider rewriting the specific test to pass if the optional is not set
    CATCH_REQUIRE(info.has_value());
    CATCH_CAPTURE(info->current_power_usage_error);
    CATCH_REQUIRE(info->current_power_usage.has_value());
    CATCH_CHECK(*info->current_power_usage > 0);
}

DEFINE_TEST_FIXTURE_METHOD(
        "get_device_status_info with valid device returns with non-zero power_cap") {
    if (!first_accessible_device) {
        return;
    }
    auto info = get_device_status_info(*first_accessible_device);

    // N.B. test may fail has_value() check if a CI runner GPU does not support an nvml query
    // in which case consider rewriting the specific test to pass if the optional is not set
    CATCH_REQUIRE(info.has_value());
    CATCH_CAPTURE(info->default_power_cap_error);
    CATCH_REQUIRE(info->default_power_cap.has_value());
    CATCH_CHECK(*info->default_power_cap > 0);
}

DEFINE_TEST_FIXTURE_METHOD(
        "get_device_status_info with valid device returns with percentage_utilization_gpu in "
        "range") {
    if (!first_accessible_device) {
        return;
    }
    auto info = get_device_status_info(*first_accessible_device);

    // N.B. test may fail has_value() check if a CI runner GPU does not support an nvml query
    // in which case consider rewriting the specific test to pass if the optional is not set
    CATCH_REQUIRE(info.has_value());
    CATCH_CAPTURE(info->percentage_utilization_error);
    CATCH_REQUIRE(info->percentage_utilization_gpu.has_value());
    CATCH_CHECK(*info->percentage_utilization_gpu <= 100);
}

DEFINE_TEST_FIXTURE_METHOD(
        "get_device_status_info with valid device returns with percentage_utilization_memory in "
        "range") {
    if (!first_accessible_device) {
        return;
    }
    auto info = get_device_status_info(*first_accessible_device);

    // N.B. test may fail has_value() check if a CI runner GPU does not support an nvml query
    // in which case consider rewriting the specific test to pass if the optional is not set
    CATCH_REQUIRE(info.has_value());
    CATCH_CAPTURE(info->percentage_utilization_error);
    CATCH_REQUIRE(info->percentage_utilization_memory.has_value());
    CATCH_CHECK(*info->percentage_utilization_memory <= 100);
}

DEFINE_TEST_FIXTURE_METHOD(
        "get_device_status_info with valid device returns with current_performance_state in "
        "range") {
    if (!first_accessible_device) {
        return;
    }
    auto info = get_device_status_info(*first_accessible_device);

    // N.B. test may fail has_value() check if a CI runner GPU does not support an nvml query
    // in which case consider rewriting the specific test to pass if the optional is not set
    CATCH_REQUIRE(info.has_value());
    CATCH_CAPTURE(info->current_performance_state_error);
    CATCH_REQUIRE(info->current_performance_state.has_value());
    CATCH_CHECK(*info->current_performance_state <= 32);
}

DEFINE_TEST_FIXTURE_METHOD(
        "get_device_status_info with valid device returns with current_throttling_reason in "
        "range") {
    if (!first_accessible_device) {
        return;
    }
    auto info = get_device_status_info(*first_accessible_device);

    // N.B. test may fail has_value() check if a CI runner GPU does not support an nvml query
    // in which case consider rewriting the specific test to pass if the optional is not set
    CATCH_REQUIRE(info.has_value());
    CATCH_CAPTURE(info->current_throttling_reason_error);
    CATCH_REQUIRE(info->current_throttling_reason.has_value());
    CATCH_CHECK(*info->current_throttling_reason <= 0x1000ULL);
}

DEFINE_TEST_FIXTURE_METHOD(
        "get_device_status_info with valid device returns with non-empty device name") {
    if (!first_accessible_device) {
        return;
    }
    auto info = get_device_status_info(*first_accessible_device);

    // N.B. test may fail has_value() check if a CI runner GPU does not support an nvml query
    // in which case consider rewriting the specific test to pass if the optional is not set
    CATCH_REQUIRE(info.has_value());
    CATCH_CAPTURE(info->device_name_error);
    CATCH_REQUIRE(info->device_name.has_value());
    CATCH_CHECK_FALSE(info->device_name->empty());
}

DEFINE_TEST_FIXTURE_METHOD(
        "get_devices_status_info returns at least one entry (with device name assigned) "
        "if there is an accessible device") {
    if (!first_accessible_device) {
        return;
    }
    CATCH_CAPTURE(*first_accessible_device);
    auto info_collection = get_devices_status_info();
    CATCH_REQUIRE_FALSE(info_collection.empty());
    CATCH_REQUIRE(info_collection.size() > *first_accessible_device);
    CATCH_REQUIRE(info_collection[*first_accessible_device].has_value());
    auto info = *info_collection[*first_accessible_device];
    CATCH_CAPTURE(info.device_name_error);

    // N.B. test may fail has_value() check if a CI runner GPU does not support an nvml query
    // in which case consider rewriting the specific test to pass if the optional is not set
    CATCH_REQUIRE(info.device_name.has_value());
    CATCH_CHECK_FALSE(info.device_name->empty());
}

#if !defined(__APPLE__)
DEFINE_TEST("get_device_count returns a non zero value if torch getNumGPUs is non-zero") {
    if (!torch::getNumGPUs()) {
        return;
    }
    CATCH_CHECK(get_device_count() > 0);
}
#else
DEFINE_TEST("get_device_count on apple returns zero") { CATCH_REQUIRE(get_device_count() == 0); }
#endif

}  // namespace dorado::utils::gpu_monitor::test
