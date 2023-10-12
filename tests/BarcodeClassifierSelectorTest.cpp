#include "read_pipeline/BarcodeClassifierSelector.h"

#include <catch2/catch.hpp>

#define TEST_GROUP "[dorado::demux::BarcodeClassifierSelector]"

namespace {

TEST_CASE(TEST_GROUP " constructor does not throw", TEST_GROUP) {
    REQUIRE_NOTHROW(dorado::demux::BarcodeClassifierSelector{});
}

TEST_CASE(TEST_GROUP " get_barcoder with valid kit does not throw", TEST_GROUP) {
    dorado::demux::BarcodeClassifierSelector cut{};

    REQUIRE_NOTHROW(cut.get_barcoder("SQK-RAB201"));
}

TEST_CASE(TEST_GROUP " get_barcoder with invalid kit throws", TEST_GROUP) {
    dorado::demux::BarcodeClassifierSelector cut{};

    REQUIRE_THROWS(cut.get_barcoder("ABSOLUTE-RUBBISH"));
}

TEST_CASE(TEST_GROUP " get_barcoder twice with same kit returns same barcoder instance",
          TEST_GROUP) {
    dorado::demux::BarcodeClassifierSelector cut{};

    auto & barcoder_first = cut.get_barcoder("SQK-RAB201");
    auto & barcoder_second = cut.get_barcoder("SQK-RAB201");

    REQUIRE(&barcoder_first == &barcoder_second);
}

TEST_CASE(TEST_GROUP " get_barcoder twice with different kits returns different barcoder instances",
          TEST_GROUP) {
    dorado::demux::BarcodeClassifierSelector cut{};

    auto & barcoder_first = cut.get_barcoder("SQK-RAB201");
    auto & barcoder_second = cut.get_barcoder("SQK-LWB001");

    REQUIRE(&barcoder_first != &barcoder_second);
}

}  // namespace