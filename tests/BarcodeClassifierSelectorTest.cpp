#include "demux/BarcodeClassifierSelector.h"

#include "demux/barcoding_info.h"

#include <catch2/catch_all.hpp>

#define TEST_GROUP "[dorado::demux::BarcodeClassifierSelector]"

namespace {

TEST_CASE(TEST_GROUP " constructor does not throw", TEST_GROUP) {
    REQUIRE_NOTHROW(dorado::demux::BarcodeClassifierSelector{});
}

TEST_CASE(TEST_GROUP " get_barcoder with valid kit does not throw", TEST_GROUP) {
    dorado::demux::BarcodeClassifierSelector cut{};

    dorado::demux::BarcodingInfo info;
    info.kit_name = "SQK-RAB201";
    REQUIRE_NOTHROW(cut.get_barcoder(info));
}

TEST_CASE(TEST_GROUP " get_barcoder with invalid kit throws", TEST_GROUP) {
    dorado::demux::BarcodeClassifierSelector cut{};

    dorado::demux::BarcodingInfo info;
    info.kit_name = "ABSOLUTE-RUBBISH";
    REQUIRE_THROWS(cut.get_barcoder(info));
}

TEST_CASE(TEST_GROUP " get_barcoder twice with same kit returns same barcoder instance",
          TEST_GROUP) {
    dorado::demux::BarcodeClassifierSelector cut{};

    dorado::demux::BarcodingInfo info;
    info.kit_name = "SQK-RAB201";
    auto barcoder_first = cut.get_barcoder(info);

    dorado::demux::BarcodingInfo info2;
    info2.kit_name = "SQK-RAB201";
    auto barcoder_second = cut.get_barcoder(info2);

    REQUIRE(barcoder_first == barcoder_second);
}

TEST_CASE(TEST_GROUP " get_barcoder twice with different kits returns different barcoder instances",
          TEST_GROUP) {
    dorado::demux::BarcodeClassifierSelector cut{};

    dorado::demux::BarcodingInfo info;
    info.kit_name = "SQK-RAB201";
    auto barcoder_first = cut.get_barcoder(info);

    dorado::demux::BarcodingInfo info2;
    info2.kit_name = "SQK-LWB001";
    auto barcoder_second = cut.get_barcoder(info2);

    REQUIRE(barcoder_first != barcoder_second);
}

}  // namespace
