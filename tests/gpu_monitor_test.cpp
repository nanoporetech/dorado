#include "utils/gpu_monitor.h"

#include <torch/torch.h>
// Catch2 must come after torch since it defines CHECK()
#include <catch2/catch.hpp>

#include <algorithm>
#include <cctype>

#define CUT_TAG "[dorado::utils::gpu_monitor]"
#define DEFINE_TEST(name) TEST_CASE(CUT_TAG " " name, CUT_TAG)

namespace {

DEFINE_TEST("get_nvidia_driver_version has value if torch::hasCUDA") {
    auto driver_version = dorado::utils::gpu_monitor::get_nvidia_driver_version();
    if (torch::hasCUDA()) {
        REQUIRE(driver_version.has_value());
    }
}

DEFINE_TEST("get_nvidia_driver_version retruns valid version string") {
    auto driver_version = dorado::utils::gpu_monitor::get_nvidia_driver_version();
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

DEFINE_TEST("get_nvidia_driver_version multiple calls return the same result") {
    auto driver_version_0 = dorado::utils::gpu_monitor::get_nvidia_driver_version();
    auto driver_version_1 = dorado::utils::gpu_monitor::get_nvidia_driver_version();
    CHECK(driver_version_0.has_value() == driver_version_1.has_value());
    if (driver_version_0.has_value()) {
        REQUIRE(*driver_version_0 == *driver_version_1);
    }
}

#if defined(__APPLE__)
DEFINE_TEST("get_nvidia_driver_version does not have value on Apple") {
    auto driver_version = dorado::utils::gpu_monitor::get_nvidia_driver_version();
    CHECK(!driver_version.has_value());
}
#endif  // __APPLE__

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
        CAPTURE(test.test_name);
        auto version = dorado::utils::gpu_monitor::detail::parse_nvidia_version_line(test.line);
        CHECK(version.has_value() == test.valid);
        if (version.has_value() && test.valid) {
            CHECK(version == test.version);
        }
    }
}

#if !defined(__APPLE__)
DEFINE_TEST("get_device_count does not throw") {
    REQUIRE_NOTHROW(dorado::utils::gpu_monitor::detail::get_device_count());
}

DEFINE_TEST("get_device_count returns a value if torch getNumGPUs is non-zero") {
    if (!torch::getNumGPUs()) {
        return;
    }
    auto num_devices = dorado::utils::gpu_monitor::detail::get_device_count();
    REQUIRE(num_devices.has_value());
}

DEFINE_TEST("get_device_count returns a non zero value if torch getNumGPUs is non-zero") {
    if (!torch::getNumGPUs()) {
        return;
    }
    auto num_devices = dorado::utils::gpu_monitor::detail::get_device_count();
    CHECK(num_devices.has_value());
    REQUIRE(*num_devices > 0);
}
#endif  //#if !defined(__APPLE__)

}  // namespace
