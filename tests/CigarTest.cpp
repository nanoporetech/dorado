#include "utils/cigar.h"

#include "TestUtils.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#define TEST_GROUP "[cigar]"

namespace dorado {
namespace cigartests {

CATCH_TEST_CASE("CigarTest: parse_cigar_from_string - empty input", TEST_GROUP) {
    const std::string_view in_cigar{""};
    const std::vector<CigarOp> expected{};
    const std::vector<CigarOp> result = parse_cigar_from_string(in_cigar);
    CATCH_CHECK(expected == result);
}

CATCH_TEST_CASE("CigarTest: parse_cigar_from_string - non-cigar", TEST_GROUP) {
    const std::string_view in_cigar{"wrong"};
    // clang-format off
    const std::vector<CigarOp> expected{
        {CigarOpType::UNDEFINED, 0},
        {CigarOpType::UNDEFINED, 0},
        {CigarOpType::UNDEFINED, 0},
        {CigarOpType::UNDEFINED, 0},
        {CigarOpType::UNDEFINED, 0},
    };
    // clang-format on
    const std::vector<CigarOp> result = parse_cigar_from_string(in_cigar);
    CATCH_CHECK(expected == result);
}

CATCH_TEST_CASE("CigarTest: parse_cigar_from_string - all ops", TEST_GROUP) {
    const std::string_view in_cigar{"1M2I3D4N5S6H7P8=9X10Y11m12x13i14d"};
    // clang-format off
    const std::vector<CigarOp> expected{
        {CigarOpType::M, 1},
        {CigarOpType::I, 2},
        {CigarOpType::D, 3},
        {CigarOpType::N, 4},
        {CigarOpType::S, 5},
        {CigarOpType::H, 6},
        {CigarOpType::P, 7},
        {CigarOpType::EQ, 8},
        {CigarOpType::X, 9},
        {CigarOpType::UNDEFINED, 10},
        {CigarOpType::UNDEFINED, 11},
        {CigarOpType::UNDEFINED, 12},
        {CigarOpType::UNDEFINED, 13},
        {CigarOpType::UNDEFINED, 14},
    };
    // clang-format on
    const std::vector<CigarOp> result = parse_cigar_from_string(in_cigar);
    CATCH_CHECK(expected == result);
}

CATCH_TEST_CASE("CigarTest: convert_mm2_cigar - empty input", TEST_GROUP) {
    const std::vector<uint32_t> in_cigar{};

    const std::vector<CigarOp> expected{};
    const std::vector<CigarOp> result =
            convert_mm2_cigar(std::data(in_cigar), static_cast<uint32_t>(std::size(in_cigar)));
    CATCH_CHECK(expected == result);
}

CATCH_TEST_CASE("CigarTest: convert_mm2_cigar - all combos", TEST_GROUP) {
    // Note: lower 4 bits encode the CIGAR operation, and only values 0-9 (non-inclusive) are used.
    // The rest are undefined.
    const std::vector<uint32_t> in_cigar{
            0x10, 0x21, 0x32, 0x43, 0x54, 0x65, 0x76, 0x87,
            0x98, 0xA9, 0xBA, 0xCB, 0xDC, 0xED, 0xFE, 0x10F,
    };

    // clang-format off
    const std::vector<CigarOp> expected{
        {CigarOpType::M, 1},
        {CigarOpType::I, 2},
        {CigarOpType::D, 3},
        {CigarOpType::N, 4},
        {CigarOpType::S, 5},
        {CigarOpType::H, 6},
        {CigarOpType::P, 7},
        {CigarOpType::EQ, 8},
        {CigarOpType::X, 9},
        {CigarOpType::UNDEFINED, 10},
        {CigarOpType::UNDEFINED, 11},
        {CigarOpType::UNDEFINED, 12},
        {CigarOpType::UNDEFINED, 13},
        {CigarOpType::UNDEFINED, 14},
        {CigarOpType::UNDEFINED, 15},
        {CigarOpType::UNDEFINED, 16},
    };
    // clang-format on

    const std::vector<CigarOp> result =
            convert_mm2_cigar(std::data(in_cigar), static_cast<uint32_t>(std::size(in_cigar)));
    CATCH_CHECK(expected == result);
}

CATCH_TEST_CASE("CigarTest: serialize_cigar - empty input", TEST_GROUP) {
    const std::vector<CigarOp> in_cigar{};

    const std::string_view expected{""};

    // Check that writing to stream works fine.
    CATCH_SECTION("Writing to a stream") {
        // Run unit under test.
        std::ostringstream oss;
        oss << in_cigar;
        const std::string result = oss.str();

        CATCH_CHECK(expected == result);
    }

    // Check that the conversion to string works fine.
    CATCH_SECTION("Conversion to std::string") {
        // Run unit under test.
        const std::string result = serialize_cigar(in_cigar);

        CATCH_CHECK(expected == result);
    }
}

CATCH_TEST_CASE("CigarTest: serialize_cigar - normal input", TEST_GROUP) {
    // clang-format off
    const std::vector<CigarOp> in_cigar{
        {CigarOpType::M, 1},
        {CigarOpType::I, 2},
        {CigarOpType::D, 3},
        {CigarOpType::N, 4},
        {CigarOpType::S, 5},
        {CigarOpType::H, 6},
        {CigarOpType::P, 7},
        {CigarOpType::EQ, 8},
        {CigarOpType::X, 9},
    };
    // clang-format on

    const std::string_view expected{"1M2I3D4N5S6H7P8=9X"};

    // Check that writing to stream works fine.
    CATCH_SECTION("Writing to a stream") {
        // Run unit under test.
        std::ostringstream oss;
        oss << in_cigar;
        const std::string result = oss.str();

        CATCH_CHECK(expected == result);
    }

    // Check that the conversion to string works fine.
    CATCH_SECTION("Conversion to std::string") {
        // Run unit under test.
        const std::string result = serialize_cigar(in_cigar);

        CATCH_CHECK(expected == result);
    }
}

CATCH_TEST_CASE("CigarTest: serialize_cigar - malformed input, unknown ops", TEST_GROUP) {
    // IMPORTANT: Malformed CIGAR ops do not throw by design for the sake of speed (simple lookup table is used).

    // clang-format off
    const std::vector<CigarOp> in_cigar{
        {CigarOpType::M, 1},
        {CigarOpType::I, 2},
        {CigarOpType::D, 3},
        {CigarOpType::UNDEFINED, 4},
        {CigarOpType::UNDEFINED, 5},
        {CigarOpType::UNDEFINED, 6},
        {CigarOpType::EQ, 7},
        {CigarOpType::X, 8},
    };
    // clang-format on

    const std::string_view expected{"1M2I3D4U5U6U7=8X"};

    // Check that writing to stream works fine.
    CATCH_SECTION("Writing to a stream") {
        // Run unit under test.
        std::ostringstream oss;
        oss << in_cigar;
        const std::string result = oss.str();

        CATCH_CHECK(expected == result);
    }

    // Check that the conversion to string works fine.
    CATCH_SECTION("Conversion to std::string") {
        // Run unit under test.
        const std::string result = serialize_cigar(in_cigar);

        CATCH_CHECK(expected == result);
    }
}

}  // namespace cigartests
}  // namespace dorado
